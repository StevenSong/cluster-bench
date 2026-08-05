# Context for next session: cluster-bench

## What this repo is
`~/repos/cluster-bench` — a **benchmarking / acceptance suite for new cluster
nodes**. It began as a Gemma-4-31B medical SFT demo; the demo was removed in the
Tier 2 refactor (still in git history before that commit) and the repo is now
only the benchmark.

Key modules under `src/cluster_bench/`: `config.py` (RunSpec — every knob),
`strategies.py` (Matrix 2A), `placement.py` (Matrix 2B + cell validation),
`ds_config.py` / `accel_config.py` (generated launcher configs),
`metrics.py`, `provenance.py`, `train.py` (one cell), `sweep.py` (matrix
driver, SSH from node0), `report.py`. NCCL tuning lives in `configs/env.sh`;
matrices in `configs/matrix/*.yaml`; one JSON per cell in `results/runs/`.

README.md carries the full operating manual. This file is the standing plan.

## Hardware
- 2 nodes × 4 H200 (141 GB each)
- GPUs in **NVLinked pairs**, 2×2 per node (NVLink domain = 2 GPUs, no NVSwitch)
- Each pair has its own rail-aligned NIC; RoCEv2 (not IB)
- Three-tier interconnect: NVLink (in-pair) >> PCIe (cross-pair, same node) ≈ RoCE (cross-node)
- Open question driving everything: **is the cross-pair PCIe hop worse than the network?**

## Status
- Tier 0 (topology validation, nccl-tests, rail/ring verification) — DONE
- Tier 1 (NCCL env var sweep) — DONE
- **Tier 2 (sharding strategy) — this is the current work**

## Proxy model: dense ~4B (candidate: Qwen3.5-4B)
**Verify before starting: dense not MoE, layer count, hidden size, vocab size, tied embeddings.**
If it's MoE the study is invalid as a dense-31B proxy (expert all-to-all is a different
comm pattern). Any dense ~4B substitutes with no other plan changes.

### Why a 4B proxy is valid
ZeRO-3 comm ≈ 6P bytes/GPU/step; compute ≈ 8·P·tokens FLOPs. **P cancels** — the
comm:compute ratio depends on *tokens/GPU/step*, not model size.

### THE critical control
Pin `per_device_train_batch_size=2`, `max_length=4096` → **8192 tokens/GPU/step**,
matching the 31B operating point — even though it wastes most of the card. If you let
4B run at its natural micro-batch (16+) everything becomes compute-bound and all
differences vanish into noise.

### What 4B cannot tell you
1. **hpZ memory tradeoff** — costs +4 GB at 4B (free), +31 GB at 31B (a real decision)
2. **Large-message efficiency** — 4B gathers ~28 MB/rank vs several hundred MB at 31B;
   sits closer to latency-bound, so it mildly **overstates hpZ=2's advantage**

### Bonus 4B enables
Plain **DDP fits** (8+8+48 = 64 GB/GPU) → a measured **zero-param-comm compute ceiling**.
Report all overheads against this, not against spec FLOPS.

## Matrix 2A — sharding configs (8 GPUs, 8192 tok/GPU/step)
| # | Config | Param gather | Grad reduce | Purpose |
|---|---|---|---|---|
| 1 | DDP | none | global | **compute ceiling / denominator** |
| 2 | ZeRO-1 | none | global | optimizer sharding only |
| 3 | ZeRO-2 | none | global | + grad sharding |
| 4 | ZeRO-3 flat (hpZ=0) | global | global | current baseline |
| 5 | ZeRO-3 hpZ=4 | node-local | global | no cross-node gathers |
| 6 | ZeRO-3 hpZ=2 | pair-local (NVLink) | global | no PCIe/RoCE gathers |
| 7 | FSDP2 FULL_SHARD | global | global | cross-check vs #4 |
| 8 | FSDP2 HYBRID_SHARD | node-local | global | cross-check vs #5 |
| 9 | TP=2 × ZeRO-3(4) | in-pair | global | only sane TP degree here |

Rows 1→2→3→4 decompose ZeRO's cost: optimizer, then grads, then params.

**Interpretive guard:** hpZ optimizes *only* the param all-gather. Grad reduce-scatter
stays global (crosses RoCE) in every config. Gathers ≈ 14 of ~21 GB/step, so **hpZ's
benefit is capped at ~2/3 of ZeRO-3's comm**. A larger measured win means something's wrong.

## Matrix 2B — placement study (top 3 configs from 2A)
Pin via `CUDA_VISIBLE_DEVICES` + host list.

| Placement | GPUs | Links |
|---|---|---|
| Within pair | 2 | NVLink only |
| Across pairs | 2 | **PCIe only** |
| Across nodes | 2 | RoCE only |
| One node | 4 | NVLink + PCIe |
| **One pair per node** | 4 | **NVLink + RoCE, zero PCIe** |
| Full | 8 | everything |

Row 5 vs row 4 is the sharp comparison — a clean 4-GPU-vs-4-GPU answer to the
PCIe-vs-network question. Only possible with a model this small.

## Matrix 2C — tokens/GPU crossover
Top 3 configs × {2048, 4096, 8192, 16384, 32768} tok/GPU/step (vary micro-batch,
fixed 4096 seq len). Finds where comm stops mattering. **Most durable output** — a
portable cluster characteristic that transfers to un-benchmarked models.

## Data: real corpus, `wrapped` packing (changed 2026-08-05)
Timing runs use the real dataset, not synthetic. The control survives because
`packing_strategy: wrapped` cuts the tokenized corpus into chunks of *exactly*
`max_length`, so `micro_batch × seq_len` is the literal token count. **Never
switch timing runs to `bfd`** — it emits sequences of at most `max_length` and
tokens/step drifts between cells, which silently contaminates every overhead
number. `metrics` counts the tokens that reach the model and `report.check_token_control`
fails any cell drifting >1% from nominal.

Packed sequences cross document boundaries and only flash-attention's varlen
path honours the boundary, so `attn_impl` defaults to `flash_attention_2` and
**flash-attn is back as a dependency** — in `requirements-flash.txt`, a separate
`pip install ... --no-build-isolation` step because it imports torch at build
time. Every node needs it; `sweep.py`'s preflight checks each peer, and
`provenance` records the version so `report.py` catches two nodes on different
kernels.

`--attn-impl sdpa` is a supported fallback: FLOPs are identical so step times are
unaffected and the gate still compares cells to each other, but absolute loss is
not a training loss. Recorded per run as
`dataset.packed_attention_crosses_documents`.

`--dataset synthetic` remains for node bring-up and for isolating a data-path
problem from a comm problem.

## Config hygiene
- **Pin** `reduce_bucket_size`, `stage3_prefetch_bucket_size`,
  `stage3_param_persistence_threshold` — currently `"auto"` in `ds_zero3.json`, and
  auto resolves differently at 4B vs 31B, silently breaking the comparison
- `NCCL_DEBUG=WARN` for timing runs; `wall_clock_breakdown: false` except profiling cells
- Discard first ~20 steps (NCCL channel setup + allocator warmup); measure 50–100
- Fixed seed; measure steps/sec at fixed tokens/step (bfd packing makes step count vary)
- Correctness gate: every config must match a reference loss curve at fixed seed —
  catches packing-mask and loss-normalization bugs that fast-but-wrong configs hide

## Reporting
Headline metric: `comm_overhead = (step_time_config / step_time_ddp) - 1`.
Also: tokens/sec/GPU, step **p50 and p95** (p95 catches stragglers), peak allocated mem.
Record NCCL version, driver version, git SHA per run.

## Then confirm at 31B (3 runs)
Flat ZeRO-3 + top two from 2A. Checking (a) memory is affordable at 62 → 78 → 93 GB/GPU,
(b) the speed ranking survives larger messages. If it holds, the 4B proxy is validated
and future studies stay small.

## Budget
2A: 9 cells ~40 min · 2B: 18 ~1.5 hr · 2C: 15 ~1.5 hr · 31B confirm: 3 ~1 hr
**Narrow start:** 2A rows 1, 4, 5, 6 (DDP ceiling + three hpZ variants) ≈ 20 min.

## Known limits of this cluster (document, don't retest)
- **~50B dense is the full-FT ceiling** (16 bytes/param, ~850 GB usable of 1128 GB)
- 70B full FT impossible (140 GB/GPU vs 141 capacity); 31B FSDP HYBRID_SHARD borderline
- No NVSwitch → no NVLS; RoCE → no SHARP in-network reduction
- TP > 2, EP/MoE, context-parallel > 2, deep PP: runnable but measure the topology, not
  the strategy (TP all-reduces are synchronous/unhideable, unlike ZeRO's overlappable gathers)
- 2 nodes = one hop: **validates a node pair, not the fabric**. No incast, congestion,
  multi-hop, or ECMP-collision signal. Re-run on expansion.
- Untested and probably worth it: ZeRO++ `zero_quantized_weights` / `zero_quantized_gradients`
  (both `false` today) — designed for exactly this comm-bound situation
- LoRA vs full-FT comm profiles don't cross-apply; separate matrices if both are needed

## Repo state — refactor DONE (2026-08-05)
Everything the plan needed is built:
- All knobs parameterized via CLI flags or `CB_*` env vars (`config.RunSpec`)
- Per-run JSON to `results/runs/<run_id>.json`, incl. full spec + provenance
- DeepSpeed configs generated per strategy; buckets **pinned**, no more `"auto"`
- `sweep.py` expands a matrix YAML and SSH-launches peer ranks
- `report.py` computes `comm_overhead` vs a matched DDP baseline, and applies
  the hpZ ≤2/3 plausibility guard and the loss-curve correctness gate
- `placement.validate()` skips cells that would silently degrade (e.g. hpz=4
  on 2 GPUs runs as flat ZeRO-3 but keeps the hpz4 label)

Still open / needs on-cluster verification (no GPU on the dev machine, so none
of this has executed):
- **FSDP2 hybrid**: `accel_config.py` emits `fsdp_shard_size` for the 2-D mesh.
  Confirm accelerate 1.14 honors that key for `fsdp_version: 2`; if not, rows 8
  of 2A needs the v1 `fsdp_sharding_strategy: HYBRID_SHARD` spelling instead.
- **DeepSpeed AutoTP** (`tensor_parallel.autotp_size`) for row 9 — verify 0.19.2
  supports it for *training*, not just inference.
- **`wrapped` packing on a prompt-completion dataset**: confirm trl 1.8's
  `pack_dataset(strategy="wrapped")` carries `completion_mask` through the
  concat-and-chunk and yields chunks of exactly `max_length`. The first run's
  `tokens_per_step_control_held` answers this directly — if it is false, the
  whole real-data switch is invalid and `bfd` is not the fallback (see above);
  the fallback is `--dataset synthetic`.
- **TRL pretokenized path** (synthetic escape hatch only): passes `input_ids` +
  pre-masked `labels` with `skip_prepare_dataset: True`.
- **`_BenchSFTTrainer.training_step` signature**: subclassed in `train.py` to
  count tokens; passes `*args/**kwargs` through, but confirm transformers 5.5
  still calls it positionally as `(model, inputs, num_items_in_batch)`.
- Proxy model still unverified: check Qwen3.5-4B is dense, not MoE.
  `modeling.check_dense` aborts the run if it isn't.
- `wandb`, `openai` dropped from requirements (unused by the benchmark).
  `flash-attn` was dropped too, then **restored** when timing moved to real
  packed data — see the data section above. It lives in
  `requirements-flash.txt`, not `requirements.txt`.
- **flash-attn build on the cluster**: PyPI ships an sdist, so `pip install`
  compiles kernels. Find the prebuilt wheel matching torch 2.9.1 / cu130 /
  cp312 / cxx11abi from the GitHub releases, confirm it imports on both nodes,
  then pin that exact version in `requirements-flash.txt` — the floor pin there
  is a placeholder.
