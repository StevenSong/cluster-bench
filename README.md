# cluster-bench

Acceptance and interconnect benchmarking for new cluster nodes.

The cluster is 2 nodes × 4 H200 (141 GB). GPUs are in **NVLinked pairs**, 2×2
per node, with no NVSwitch — so the NVLink domain is 2 GPUs wide. Each pair has
its own rail-aligned NIC over RoCEv2. That gives three interconnect tiers:

```
NVLink (in-pair)  >>  PCIe (cross-pair, same node)  ≈  RoCE (cross-node)
```

The open question driving everything: **is the cross-pair PCIe hop worse than
the network?** If it is, the right sharding strategy is not the obvious one.

Tier 0 (topology validation, nccl-tests) and Tier 1 (NCCL env var sweep) are
done. This repo is Tier 2: sharding strategy, placement, and the tokens/GPU
crossover.

## Quickstart

```bash
conda env create -f env.yaml && conda activate cluster-bench
```

```bash
python -m cluster_bench.sweep configs/matrix/2a_sharding.yaml --list
```

```bash
python -m cluster_bench.sweep configs/matrix/2a_sharding.yaml --hosts localhost node1
```

```bash
python -m cluster_bench.report --runs-dir results/runs
```

The env must exist on **every** node. `sweep.py` activates it explicitly on
each rank, defaulting to whichever env the driver itself is running in — peer
ranks are launched with `ssh … bash -lc`, and `conda init` puts its shell hook
in `~/.bashrc`, which returns early for non-interactive shells, so a peer would
otherwise get system Python and rank 0 would sit in the rendezvous until it
timed out. Override with `--conda-env` / `--conda-base` if the peers' install
lives elsewhere, or `--no-conda` if their shell rc already handles it. A
one-ssh-per-peer preflight runs before the first cell and fails loudly rather
than at the NCCL timeout; skip it with `--no-preflight`.

The narrow start — DDP ceiling plus the three hpZ variants, about 20 minutes:

```bash
python -m cluster_bench.sweep configs/matrix/2a_sharding.yaml --only ddp zero3 zero3-hpz4 zero3-hpz2
```

## Why a 4B proxy for a 31B target

ZeRO-3 comm is ≈ 6P bytes/GPU/step; compute is ≈ 8·P·tokens FLOPs. **P cancels.**
The comm:compute ratio depends on *tokens/GPU/step*, not on model size. So a
dense ~4B model measures the same ratio the 31B run will see — provided it is
held at the 31B operating point.

That proviso is the single most important thing in this repo:

> `--seq-len 4096 --micro-batch 2` → **8192 tokens/GPU/step**, matching 31B,
> even though it wastes most of an H200. Let a 4B model run at its natural
> micro-batch (16+) and everything becomes compute-bound and every difference
> the study is trying to measure vanishes into noise.

`RunSpec` defaults to exactly this. Don't "fix" it.

Two things the proxy cannot tell you:

1. **The hpZ memory tradeoff.** hpZ=2 keeps a second node-local copy of every
   parameter shard: +4 GB at 4B (free), +31 GB at 31B (a real decision).
2. **Large-message efficiency.** 4B gathers ~28 MB/rank against several hundred
   MB at 31B, so it sits closer to latency-bound and mildly **overstates**
   hpZ=2's advantage.

What the proxy *adds*: plain **DDP fits** at 4B (8 + 8 + 48 = 64 GB/GPU), giving
a measured zero-param-comm compute ceiling. Every overhead number here is
reported against that, not against spec FLOPS.

Before starting, verify the proxy is **dense, not MoE** — expert all-to-all is a
different comm pattern and invalidates the whole comparison. `modeling.py`
checks this on every run and aborts; it also records layer count, hidden size,
vocab size and embedding tying into the results JSON.

## The matrices

| File | What it varies | Cells | Budget |
|---|---|---|---|
| `configs/matrix/2a_sharding.yaml` | 9 sharding strategies, 8 GPUs | 9 | ~40 min |
| `configs/matrix/2b_placement.yaml` | 6 GPU placements × top 3 from 2A | ≤18 | ~1.5 hr |
| `configs/matrix/2c_tokens.yaml` | 5 token counts × top 3 from 2A | 15 | ~1.5 hr |
| `configs/matrix/gate_correctness.yaml` | loss curve on real data | 4 | — |

**2A** decomposes ZeRO's cost — `ddp → zero1 → zero2 → zero3` isolates
optimizer state, then gradients, then parameters — and then varies *only* the
param-gather scope: flat (global) → hpz4 (node-local) → hpz2 (pair-local).

**2B** is the placement study. The sharp comparison is `one-node`
(NVLink + PCIe) against `pair-per-node` (NVLink + RoCE, **zero PCIe**): same
four GPUs' worth of compute, and a direct answer to the PCIe-vs-network
question. Only possible with a model this small.

Placements in `placement.py` are written in **node0's device order**. Peer
nodes enumerate their GPUs in the opposite order, so `Placement.devices_for()`
reverses the list for every `machine_rank > 0` before it becomes
`CUDA_VISIBLE_DEVICES`. Same physical GPUs either way — what the reversal fixes
is which local rank lands on which rail-aligned NIC. Without it, cross-node
collectives ride mismatched rails and the run still completes, just slower, so
2B would be measuring the launch bug rather than the interconnect tier.

**2C** finds where comm stops mattering. This is the most durable output of the
study — a portable cluster characteristic that transfers to models nobody has
benchmarked yet.

## Reading the results

Headline metric is `comm_overhead = step_time_config / step_time_ddp - 1`,
matched to a DDP baseline in the same (placement, tokens) group. `report.py`
also prints tokens/sec/GPU, step **p50 and p95** — p95 catches stragglers a
mean would hide — and peak allocated memory.

**Interpretive guard, applied automatically.** hpZ optimizes *only* the
parameter all-gather. Gradient reduce-scatter stays global, and crosses RoCE, in
every config. Gathers are roughly 14 of ~21 GB/step, so hpZ's benefit is capped
at about **2/3** of ZeRO-3's comm. `report.py` warns when a measured win exceeds
that — a bigger number means something is wrong, usually a cell that quietly
degraded to a different config.

`placement.validate()` refuses cells that cannot mean what they claim, rather
than running them. DeepSpeed clamps `zero_hpz_partition_size` to the world size,
so `hpz=4` on 2 GPUs runs as flat ZeRO-3 while still being labelled hpz4 — those
cells are skipped with a printed reason.

Run the correctness gate before trusting any timing number. Every config must
reproduce the reference loss curve at a fixed seed; this is what catches
packing-mask and loss-normalization bugs, which otherwise present as a config
that is impressively fast and quietly wrong.

## Layout

```
src/cluster_bench/
  config.py       RunSpec — every knob, serialized into each result
  strategies.py   the 9 sharding configs of Matrix 2A
  placement.py    the 6 GPU placements of Matrix 2B + cell validation
  ds_config.py    DeepSpeed config builder (pinned buckets, see below)
  accel_config.py accelerate launch config builder
  modeling.py     model load + the dense-not-MoE pre-flight check
  data.py         real dataset (timing + gate) + a synthetic escape hatch
  metrics.py      step timing, p50/p95, tokens/s/GPU, peak memory
  provenance.py   NCCL/driver/torch versions, git SHA, NCCL env in effect
  train.py        one cell -> one results JSON
  sweep.py        matrix driver, SSH-launches peer ranks from node0
  report.py       aggregate + the interpretive guards above
configs/
  env.sh          NCCL tuning (the frozen Tier 1 result)
  matrix/*.yaml   the matrices
results/runs/     one JSON per cell
```

## Config hygiene

`reduce_bucket_size`, `stage3_prefetch_bucket_size` and
`stage3_param_persistence_threshold` are **pinned**, not `"auto"`. DeepSpeed's
auto resolves them from the live model's `hidden_size`, which means a 4B proxy
and the 31B target would move messages 4× apart in size — silently breaking the
comparison the proxy exists to make. `config.buckets_for_hidden` pins all three
to what auto *would* have produced at the target's hidden size (5120), so the
proxy moves target-sized buckets. Override with `--bucket-ref-hidden`.

Also fixed across cells: seed, `NCCL_DEBUG=WARN` for timing runs,
`wall_clock_breakdown: false`, no checkpointing, no experiment tracker. Step
time is the **max across ranks**, not rank 0's local view — every rank blocks on
the next collective anyway, so the slowest rank *is* the step time. The first 20
steps are discarded (NCCL channel setup, allocator warmup) and 80 are measured.

Timing runs and the correctness gate both use the **real dataset**, so the
tokenizer's length distribution, the packing path and the completion mask are
exercised by the numbers being reported rather than only by the gate.

Real text costs the control, and `packing_strategy` is how it is bought back.
bfd emits sequences of *at most* `max_length`, so tokens-per-step — the
denominator of every throughput number — would drift between cells. **`wrapped`**
concatenates the tokenized corpus and cuts it into chunks of *exactly*
`max_length`, so `micro_batch × seq_len` stays the literal token count. The
tokens that actually reach the model are counted per step and reported as
`tokens_per_gpu_step_measured`; `report.py` flags any cell that drifts more than
1% from nominal, and any comparison group whose cells disagree.

Each cell loads only as much of the split as its step count needs (`--dataset-num-samples`
to override) — a 40-second cell should not spend its time tokenizing 25k
examples. Sampling is the head of the split, not a shuffle, so every cell in a
matrix sees byte-identical text in identical order.

Two consequences worth knowing:

- Packed sequences cross document boundaries, and only flash-attention's varlen
  path honours the boundary. Under the default `sdpa` a token attends back into
  the previous document. FLOPs are unchanged, so step times — the thing measured
  here — are unaffected, and the gate still holds because every cell is
  contaminated identically. The absolute loss is not a training loss. Runs
  record this as `dataset.packed_attention_crosses_documents`.
- `--dataset synthetic` still generates pre-tokenized random ids of exactly
  `seq_len`. Keep it for node bring-up before the corpus is staged, and for
  separating a data-path problem from a comm problem.

## Known limits of this cluster

Documented so they don't get retested:

- **~50B dense is the full-FT ceiling** (16 bytes/param, ~850 GB usable of 1128).
  70B full FT is impossible (140 GB/GPU against 141 capacity); 31B FSDP
  HYBRID_SHARD is borderline.
- No NVSwitch → no NVLS. RoCE → no SHARP in-network reduction.
- TP > 2, EP/MoE, context-parallel > 2, and deep PP all run, but on this
  topology they measure the topology rather than the strategy. TP all-reduces
  are synchronous and unhideable, unlike ZeRO's overlappable gathers.
- **2 nodes is one hop: this validates a node pair, not a fabric.** No incast,
  congestion, multi-hop, or ECMP-collision signal. Re-run on expansion.
- ZeRO++ `zero_quantized_weights` / `zero_quantized_gradients` are wired up and
  `false`. Untested, and designed for exactly this comm-bound situation —
  probably worth a look.
- LoRA and full-FT comm profiles don't cross-apply. Separate matrices if both
  are needed.

## Confirming at 31B

Three runs — flat ZeRO-3 plus the top two from 2A — checking that (a) memory is
affordable at 62 → 78 → 93 GB/GPU and (b) the speed ranking survives larger
messages. If it holds, the 4B proxy is validated and future studies stay small.

```bash
python -m cluster_bench.train --model-path /opt/gpudata/models/google/gemma-4-31B-it --strategy zero3 --placement full --micro-batch 2 --seq-len 4096
```
