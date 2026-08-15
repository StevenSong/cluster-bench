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

## Findings — read this first

**Yes, the cross-pair PCIe hop is worse than the network.** Measured through
FSDP2 at 8192 tok/GPU/step, normalised exposed communication per GPU is 0.072
on NVLink, 0.304 over RoCE, and **0.379 across the PCIe hop** — the full
three-tier hierarchy, with PCIe last. The 4-GPU comparison is the clean one:
`pair-per-node` (NVLink + RoCE, zero PCIe) costs 0.200 against `one-node`
(NVLink + PCIe) at 0.261.

Read this through `fsdp-full`, not through DDP or ZeRO-3. DDP's exposed comm is
small enough that the 2-GPU placements' own run-to-run spread (up to 5%) swamps
the NVLink-vs-RoCE difference, and ZeRO-3's spread is wider still.

### How to train on this node

Default to **FSDP2 FULL_SHARD**, not DeepSpeed ZeRO-3: same bytes on the wire,
but ~0.17 s/step of exposed communication instead of ~0.41 (17% over the
measured DDP compute ceiling rather than 41%), plus a lower and steadier tail
(p95/p50 1.16 vs 1.55), 4× better run-to-run reproducibility, and marginally
lower peak memory. Skip hpZ and skip tensor parallelism — hpZ costs +3.2 GB/GPU
for a speedup indistinguishable from zero, and TP=2 adds ~0.17 s/step of
synchronous, unhideable all-reduce even when confined entirely to NVLink. Then
push tokens per GPU per step as high as memory allows: communication per step is
fixed regardless of batch size, so overhead collapses between 8192 and 16384
tok/GPU (17% → 6% → 2%), and above ~16k tok/GPU sharding is close to free here.
If a job doesn't need all 8 GPUs, keep it off the cross-pair PCIe hop — one
NVLink pair for 2 GPUs, one pair per node for 4 (~6% faster than four GPUs on
one node). At 8 GPUs you cross everything and ~17% is the floor.

### The 2A table, 8 GPUs at 8192 tok/GPU/step

Exposed comm is p50 minus the matched DDP p50 — the DDP cell is a *measured*
zero-param-comm compute ceiling, not spec FLOPS.

| config | p50 s | exposed comm s/step | overhead | peak GB |
|---|---|---|---|---|
| ddp (ceiling) | 0.9987 † | — | — | 38.6 |
| **fsdp-full** | **1.1647** † | **0.166 ± 0.008** | **+16.6%** | **12.0** |
| zero2 | 1.1942 | 0.195 | +19.6% | 17.7 |
| zero1 | 1.1955 | 0.197 | +19.7% | 17.7 |
| zero0 | 1.3163 | 0.318 | +31.8% | 75.0 |
| tp2-zero2 | 1.3652 | 0.366 | +36.7% | 16.4 |
| zero3 | 1.4015 † | 0.403 ± 0.023 | +40.3% | 13.2 |
| zero3-hpz2 | 1.4350 | 0.436 | +43.7% | 16.4 |
| zero3-hpz4 | 1.4636 | 0.465 | +46.6% | 14.8 |

† mean of replicates (n=2–5), as printed by `report.py`'s `replicates` section.
The rest are single samples, and DeepSpeed cells carry a ~3.5% run-to-run
spread, so margins under ~10% between them are not resolvable — `zero3`,
`tp2-zero2`, `zero3-hpz2` and `zero3-hpz4` are one indistinguishable cluster.
`fsdp-full` (~1.0%) and `ddp` (~0.9%) are stable enough to compare directly.

**58.8% ± 3.1% of ZeRO-3's exposed cost is not the fabric.** FSDP2 moves
identical bytes for 41% of the time, and beats ZeRO-1/ZeRO-2, which communicate
strictly less. It amortizes, though: `zero3 − fsdp-full` falls from ~0.24 s at
8192 tok/GPU to 0.036 s at 32768, so at a 31B-sized step the backends should
converge and the choice matters less than it does here.

### What is not established

* Everything above is a dense 4B proxy at the 31B operating point. **The 31B
  confirmation has not been run**, and the "P cancels" argument only holds at
  equal MFU — see the correction in CLAUDE.md.
* The DDP denominator is bf16-in-place while every sharded cell keeps fp32
  master weights, so no reference loss curve gates it. `zero0` was built to
  close that and failed on both counts.
* 2 nodes is one hop: this validates a node pair, not a fabric. No incast,
  congestion, multi-hop or ECMP-collision signal.

## Quickstart

```bash
conda env create -f env.yaml && conda activate cluster-bench
```

```bash
pip install -r requirements-flash.txt --no-build-isolation
pip install nvidia-nccl-cu13==2.30.7
```

flash-attn is a second command because it imports torch at build time and so
needs `--no-build-isolation`, which a requirements file cannot carry. It is not
optional: packed sequences cross document boundaries and only flash-attention's
varlen path honours the boundary. From PyPI it compiles CUDA kernels — tens of
minutes, `MAX_JOBS=4` if the node swaps — so prefer the matching prebuilt wheel
from the project's GitHub releases. Both commands run on **every** node, and
`sweep.py`'s preflight checks each peer can import it before the first cell.

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

### Two architectures that break the argument

The proxy must be **dense** and **full-attention**. `modeling.py` checks both
before the weights load and stops the run rather than produce a results file.

**Mixture of experts** breaks the numerator: expert all-to-all is a comm pattern
the dense target does not have.

**Linear attention** — gated delta, Mamba, RWKV, and the hybrids that interleave
such layers with real attention — breaks the denominator, which is harder to
notice. The fused kernels live in `flash-linear-attention` and `causal-conv1d`,
neither of which this repo installs, so transformers falls back to an eager
chunked recurrence. Comm is completely unaffected — ZeRO-3 gathers bytes and
does not care what a layer computes — but compute rises in every cell by the
same factor, and `comm_overhead` is a ratio of step times. Every overhead in
Matrix 2A gets divided toward zero while the *ranking* survives intact, and
Matrix 2C reports its crossover at a lower tokens/GPU than the truth. Nothing in
the results looks wrong.

The only symptom is a transformers warning about a "fast path" not being
available. Do not dismiss it; it means the number you are about to record is
smaller than the real one. `--no-require-full-attention` exists for measuring
such a model deliberately, and is only honest if fla and causal-conv1d are
installed on **every** node.

Before starting, verify the proxy is **dense, not MoE** — expert all-to-all is a
different comm pattern and invalidates the whole comparison. `modeling.py`
checks this on every run and aborts; it also records layer count, hidden size,
vocab size and embedding tying into the results JSON.

## The matrices

| File | What it varies | Cells | Budget |
|---|---|---|---|
| `configs/matrix/2a_sharding.yaml` | 9 sharding strategies, 8 GPUs | 9 | ~40 min |
| `configs/matrix/2b_placement.yaml` | 6 GPU placements × 3 configs | 18 | ~1.5 hr |
| `configs/matrix/2c_tokens.yaml` | 5 token counts × 3 configs | 15 | ~1.5 hr |
| `configs/matrix/gate_correctness.yaml` | loss curve on real data | 5 | — |
| `configs/matrix/2b_anomaly_profile.yaml` | `wall_clock_breakdown` on the 2-GPU cells | 6 | ~15 min |

**2A** decomposes ZeRO's cost — `ddp → zero0 → zero1 → zero2 → zero3` isolates
the runtime itself, then optimizer state, then gradients, then parameters — and
then varies *only* the param-gather scope: flat (global) → hpz4 (node-local) →
hpz2 (pair-local).

Two rows changed on 2026-08-15, after the first full pass:

* **`fsdp-hybrid` dropped.** It reported 12.0 GB peak, identical to `fsdp-full`
  to the decimal, and a node-local parameter replica cannot be free — the
  DeepSpeed equivalent costs +1.6 GB at hpz4 and +3.2 GB at hpz2 on the same
  model. accelerate 1.14 took `fsdp_shard_size` under `fsdp_version: 2` and
  silently did not build the 2-D mesh, so the row was FULL_SHARD measured twice.
  Restoring it means the FSDP1 spelling (`fsdp_sharding_strategy:
  HYBRID_SHARD`) and confirming it engaged **by peak memory going up**, not by
  the key being present in the config — which is precisely what fooled it.
* **`zero0` added.** DeepSpeed with ZeRO off: the candidate repair for the
  denominator, described under "Reading the results" below.

2B and 2C now run `ddp / zero3 / fsdp-full`. They were run against
`zero3-hpz2`, which measured slower than flat ZeRO-3 in every cell of every
matrix, through a config whose step time is mostly not the fabric.

**2B** is the placement study. The sharp comparison is `one-node`
(NVLink + PCIe) against `pair-per-node` (NVLink + RoCE, **zero PCIe**): same
four GPUs' worth of compute, and a direct answer to the PCIe-vs-network
question. Only possible with a model this small.

Read 2B on **absolute p50 at a fixed GPU count, not on the overhead column**.
Everywhere else the denominator is fixed and the ratio is the point; here each
placement is divided by its own DDP cell, so the denominator moves with the row.
In the first pass that inverted the ranking — `within-pair` showed the worst
ZeRO-3 overhead of any placement (+47.1%) only because DDP is fastest on
NVLink, while its absolute step time tied with `across-pairs`.

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

**Known bias in the headline metric: the DDP baseline runs cheaper optimizer
arithmetic than every cell it is the baseline for.** `modeling.load` builds the
model in bf16 and `mixed_precision: bf16` does not upcast an already-bf16 model,
so the DDP cell holds bf16 parameters, gradients and Adam moments and applies
the update in place. Both DeepSpeed's bf16 path and accelerate's FSDP2 path keep
an fp32 master copy of every parameter and apply the update in fp32. **Row 1 is
the only cell in the matrix on the bf16 path**, which is unfortunate, because it
is the denominator.

An Adam step over 4B parameters moves ~32 GB of memory traffic in bf16 against
~64–80 GB with fp32 masters — order 15 ms on a ~1 s step. `step_time_ddp` is
that much faster than a like-for-like baseline, so every `comm_overhead` is
biased **up** by a point or two. The bias is uniform and applies equally to
every cell, so it does not reorder 2A, and it does not touch the FSDP2-vs-ZeRO-3
cross-check in rows 7–8, which is a clean comparison between two fp32-master
backends.

No flag makes the `ddp` cell itself match. Loading fp32 so it keeps master
weights also makes its gradients fp32 and doubles its all-reduce volume,
corrupting the denominator in the one dimension this study exists to measure.
Pure-bf16 DDP has the right *wire* profile — bf16 gradients, the same as
DeepSpeed's bf16 reduce — and only its optimizer step is cheap.

**`zero0` is the other route: keep the wire profile, change the runtime.**
DeepSpeed with `stage: 0` reduces bf16 gradients exactly as the DDP cell does,
but keeps fp32 master weights like every other DeepSpeed cell — so if it costs
no parameter communication, it is a like-for-like denominator that the
correctness gate also covers, and `--baseline zero0` removes the bias above
instead of documenting it.

That "if" is the whole row, and it is not decidable from the config file. With
`bf16.enabled` and ZeRO off, DeepSpeed's engine builds a `BF16_Optimizer`, which
partitions the fp32 master weights across the DP group and all-gathers the
updated bf16 parameters at the end of every step — ZeRO-1's wire profile under a
stage-0 label, with no flag to disable it. Assuming otherwise would hide an
all-gather inside the denominator and deflate every overhead number in the
study, which is the same failure as the `fsdp-hybrid` row above.

So `report.check_baseline_candidates` reads the answer off the runs:

| `zero0` lands | Meaning | What to do |
|---|---|---|
| within ~3% of `ddp` | stage 0 really is zero-param-comm | gate it, then switch to `--baseline zero0` |
| level with `zero1` | `BF16_Optimizer` partitions and all-gathers | keep `--baseline ddp`; the row is DeepSpeed's per-step runtime floor |
| **slower than `zero1`** | stage 0 is a *heavier* ZeRO-1 | keep `--baseline ddp`; the row measures nothing usable |

**Measured 2026-08-15: the third one.** `zero0` came back at 1.316 s p50 against
`ddp`'s 0.998 and `zero1`'s 1.196 — slower than the config that shards strictly
more. `BF16_Optimizer` all-reduces the whole gradient (2P) and *then*
all-gathers the updated bf16 parameters (P), where ZeRO-1 reduce-scatters and
all-gathers for 2P total, the same volume as DDP's allreduce. ~3P against ~2P;
the measured 1.6× on exposed comm matches the 1.5× on volume. DeepSpeed's stage
0 is the most expensive non-stage-3 path it has.

It fails the correctness gate too — 0.1831 above the `zero3` reference over 100
steps, against a 0.006–0.022 spread across the rest of the fp32-master family —
so its step time is not trustworthy either, and it is not a usable measurement
of the runtime floor. **The denominator is still `ddp`, and it is still
ungated.** The row stays in 2A as the standing evidence for that, and because
the verdict is re-derived automatically on every report, which is worth having
after a DeepSpeed upgrade.

The `vs` column in the table names the denominator each row was divided by.
Groups that did not run the requested baseline fall back to the next candidate
rather than blanking, so the column is the only thing that says a row changed
reference.

Which backend keeps master weights was **measured, not assumed** — the two
accelerate backends do not agree, and the intuitive reading (that FSDP2 follows
DDP) is wrong. Three independent signals, worth knowing how to read because
nothing announces this:

* **Peak memory.** `ddp` 38.6 GB unsharded is exactly bf16 params + grads + bf16
  Adam. `fsdp-full` 12.0 and `zero3` 13.2 sit where fp32 masters put them; a
  pure-bf16 FSDP2 shard would have been ~8–9.
* **Grad norms.** A norm over bf16 gradients lands exactly on the bf16 grid
  (51.75, 39.75, 27.125, 0.9921875); one computed in fp32 essentially never
  does. Only DDP's do. `report.check_precision_class_evidence` runs this check
  automatically and warns when a run's norms contradict its declared backend.
* **Loss curves.** `fsdp-full` sits 0.006–0.015 from every DeepSpeed cell over
  100 steps, and 0.288 from DDP.

Run the correctness gate before trusting any timing number. Every config must
reproduce the reference loss curve at a fixed seed; this is what catches
packing-mask and loss-normalization bugs, which otherwise present as a config
that is impressively fast and quietly wrong.

The gate only compares cells that can mean something to each other. Three
exclusions, all reported rather than silently applied:

| Skipped | Why |
|---|---|
| Different optimizer precision | The bf16/fp32-master split above separates the curves within two optimizer steps however correct both cells are |
| Different `tp_size` | A TP group shares one batch, so sample order and `num_items_in_batch` both shift |
| Different (placement, tokens) group | A different global batch is a legitimately different curve |
| Different corpus (`dataset.num_samples`) | `dataset_num_samples: 0` sizes the corpus from warmup + measure steps, so a matrix with a different step budget reshuffles the batch order |

That last one is the subtle one, and it is why `gate_correctness.yaml`'s
`warmup_steps: 0 / measure_steps: 60` cells cannot be gated against 2A's
`20 / 80` cells: same seed, same byte-identical head-of-split text, but
`len(dataset)` differs and HF's seeded `RandomSampler` permutes over the dataset
length. It presents as every cell in the matrix failing the gate by the same
~0.17, which looks like a systematic bug and is only a different batch order.

`warmup_steps` is a measurement-discard count, not an LR warmup, and `on_log`
records the curve from step 1 regardless — so the loss curve does not need
`warmup_steps: 0`, and 2A's own runs are usable as gate references directly. A
full check is then one pass per family with no dedicated gate cells at all:

```bash
python -m cluster_bench.report --reference zero3__full__t8192__s4096
```

That one pass covers every fp32-master cell in 2A — `zero0`, `zero1`, `zero2`,
`zero3`, both hpZ rows and `fsdp-full`. `tp2-zero2` is excluded on `tp_size`,
and `ddp` is the only bf16-in-place cell in the matrix, so it has nothing to be
gated against: giving it a peer means running a second DDP cell at the same step
budget, or retiring it as the denominator in favour of `zero0`. That repeat
also measures the run-to-run noise floor, which is what `--loss-tol` should be
set from — the fp32-master family currently spans 0.006–0.022 pairwise, so the
default 0.02 sits just inside the noise and flags a few cells spuriously.

Each pass ends with a line stating how many cells it actually compared. A gate
that looks like it ran but checked nothing is the failure mode it exists to
prevent.

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
  path honours the boundary — which is why `attn_impl` defaults to
  `flash_attention_2` and flash-attn is a required install. `--attn-impl sdpa`
  is a supported fallback if a node can't build it: FLOPs are unchanged, so step
  times — the thing measured here — are unaffected, and the gate still holds
  because every cell is contaminated identically. But a token attends back into
  the previous document, so the absolute loss is no longer a training loss. Runs
  record this as `dataset.packed_attention_crosses_documents`, and `report.py`
  warns if flash-attn versions differ across cells.
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
