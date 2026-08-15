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
- **Tier 2 (sharding strategy) — first full pass of 2A/2B/2C measured
  2026-08-15; second pass in flight (see Results). 31B confirm not started.**

## Results of the first pass (2026-08-15) — read this before planning runs
Convert overhead back to **exposed comm seconds/step** (p50 minus matched DDP
p50); the ratio hides the finding. At full/8192: fsdp-full 0.154, zero2 0.196,
zero1 0.198, tp2-zero2 0.367, hpz2 0.437, zero3 0.456, hpz4 0.466.

1. **~2/3 of flat ZeRO-3's cost is not the fabric.** FSDP2 FULL_SHARD moves the
   same bytes as ZeRO-3 for a third of the time, and beats ZeRO-1/ZeRO-2, which
   communicate strictly less. The residual is DeepSpeed's stage-3 runtime.
   Whether it amortizes at 31B (an 8× longer step) is now the central question.
2. **hpZ is dead.** hpz2 is slower than flat ZeRO-3 in *every* cell of 2A, 2B
   and 2C; hpz4 too. Not a config that failed to engage — memory is +3.2/+1.6 GB,
   exactly the secondary-partition arithmetic. It follows from (1): hpZ
   optimizes the param gather, and the gather is not the cost. Do not take hpZ
   to 31B; +3.2 GB at 4B extrapolates to **+25 GB/GPU at 31B**.
3. **Cross-pair PCIe is worse than the network** — the question driving the
   whole repo. Read it off the DDP rows, which are a clean link probe (one
   allreduce, same volume). 2 GPUs: NVLink 0.9097 < RoCE 0.9293 < **PCIe
   0.9759**. within-pair vs across-pairs is perfectly controlled (same node,
   same ranks) and the PCIe hop costs 7.3%. 4 GPUs: pair-per-node 0.9581 beats
   one-node 1.0123 by 5.7%. Same direction in the ZeRO-3 rows.
4. **2C crossover: between 8192 and 16384 tok/GPU/step.** Exposed comm is flat
   at ~0.44 s/step from 2048 to 8192 (comm bytes don't depend on batch size),
   then collapses to 0.195 at 16384 and 0.094 at 32768. Portable form: *this
   cluster exposes ~0.45 s/step of un-hidden ZeRO-3 traffic for a 4B model, and
   it stops mattering once compute per step exceeds ~1.5–2 s.* Trust 8192 and
   up — DDP goes 0.7236 → 0.7690 s for 2× the tokens at the bottom of the
   range, so those points measure fixed cost against fixed cost.
5. **TP=2 costs ~0.17 s/step** even entirely on NVLink (0.367 against zero2's
   0.196), as predicted: TP all-reduces are synchronous and unhideable.
6. **Unexplained: the 2-GPU same-node ZeRO-3 cells.** Exposed comm normalised
   by the (N−1)/N gather factor is 0.48–0.52 everywhere except across-pairs
   (0.71) and within-pair (0.86) — the fastest link, worst per byte. Not noise,
   not memory pressure. `configs/matrix/2b_anomaly_profile.yaml` probes it.
   Until it is explained, placement conclusions rest on the DDP rows and the
   4-GPU pair.

**Correction to the proxy argument.** "P cancels" holds for the bytes, but
comm/compute = 6·A/(8·T·B) and *achieved* FLOP/s A does not cancel. The DDP
ceiling here is ~250–260 TFLOP/s at 8192 tok/GPU (~25% MFU); 31B at hidden 5120
should reach 40–50%, which pushes overheads **up** by up to ~1.7×, partly offset
by the large-message bandwidth gain (28 MB → ~220 MB per rank) pushing down.
Both corrections are large and are being assumed to cancel. That makes the 31B
confirm the only test of the proxy, not a formality.

## Proxy model: dense, full-attention ~4B (Qwen3-4B)
**Verify before starting: dense not MoE, full attention not hybrid, layer count, hidden
size, vocab size, tied embeddings.** Two architectures invalidate the proxy, and
`modeling.py` aborts the run on each:
- **MoE** — expert all-to-all is a different comm pattern with no counterpart in the
  dense target. Breaks the numerator.
- **Linear attention / hybrid** — breaks the *denominator*, and does it silently.
  fla/causal-conv1d aren't installed, so those layers run eagerly and inflate compute
  in every cell equally; `comm_overhead` is a ratio, so every number in 2A shrinks
  toward zero and 2C's crossover lands below the truth. Comm is untouched, so nothing
  looks wrong in the results.

Any dense full-attention ~4B substitutes with no other plan changes.

**Was Qwen3.5-4B** (changed 2026-08-05): it is a gated-delta hybrid, and it announced
itself only as a transformers warning about an unavailable "fast path". Qwen3-4B is
dense full attention at the same hidden size (2560), so `bucket_ref_hidden` and every
other pinned number carry over unchanged.

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
| 2 | ZeRO-0 | none | global | like-for-like fp32-master denominator candidate |
| 3 | ZeRO-1 | none | global | optimizer sharding only |
| 4 | ZeRO-2 | none | global | + grad sharding |
| 5 | ZeRO-3 flat (hpZ=0) | global | global | current baseline |
| 6 | ZeRO-3 hpZ=4 | node-local | global | no cross-node gathers |
| 7 | ZeRO-3 hpZ=2 | pair-local (NVLink) | global | no PCIe/RoCE gathers |
| 8 | FSDP2 FULL_SHARD | global | global | cross-check vs #5 |
| 9 | TP=2 × ZeRO-2(4) | in-pair (TP, not ZeRO) | global | only sane TP degree here |

Rows 1→3→4→5 decompose ZeRO's cost: optimizer, then grads, then params.

**Row 8 was FSDP2 HYBRID_SHARD; dropped 2026-08-15.** It reported 12.0 GB peak,
identical to FULL_SHARD to the decimal, where a node-local replica has to cost
memory (hpZ=4 costs +1.6 GB, hpZ=2 +3.2 GB on the same model). accelerate 1.14
accepted `fsdp_shard_size` under `fsdp_version: 2` and did not build the 2-D
mesh, so the row measured FULL_SHARD twice. Restoring it needs the FSDP1
spelling (`fsdp_sharding_strategy: HYBRID_SHARD`) and verification **by peak
memory rising**, not by the key being in the config. Low priority: its job was
cross-checking hpZ, and hpZ is dead (below).

**Row 2 added 2026-08-15** to repair the denominator. DDP is the only
bf16-in-place cell — it holds bf16 params, grads and Adam moments — so it is
both the denominator and the one cell no reference loss curve can gate.
DeepSpeed stage 0 has DDP's wire profile with fp32 master weights, which would
fix both. **But it may not be zero-param-comm**: with `bf16.enabled` and ZeRO
off, deepspeed builds a `BF16_Optimizer` that partitions the fp32 master
weights across the DP group and all-gathers the updated bf16 params every step
— ZeRO-1's pattern under a stage-0 label, with no flag to disable it. Do not
assume either way; `report.check_baseline_candidates` decides from the data.
Within ~3% of DDP → switch to `--baseline zero0`. Level with ZeRO-1 → keep DDP
and read the row as DeepSpeed's fixed per-step runtime floor, which is worth
having on its own (see Results below).

**Row 9 is ZeRO-2, not ZeRO-3** (changed 2026-08-14): deepspeed 0.19.2 asserts
`zero_optimization_stage() <= 2` whenever autotp is on. It costs the row nothing —
TP keeps params permanently sharded in-pair, so stage 3 had no param all-gather left
to optimize; its comm is a per-layer *activation* all-reduce on NVLink plus a grad
reduce over the DP group of 4, which still crosses PCIe and RoCE. Read it against
row 3 (plain ZeRO-2): single variable, TP added. v0.19.4 (2026-08-06) removed the
assert, but taking that upgrade means re-running all of 2A on a new runtime.

**Interpretive guard:** hpZ optimizes *only* the param all-gather. Grad reduce-scatter
stays global (crosses RoCE) in every config. Gathers ≈ 14 of ~21 GB/step, so **hpZ's
benefit is capped at ~2/3 of ZeRO-3's comm**. A larger measured win means something's wrong.

## Matrix 2B — placement study (`ddp / zero3 / fsdp-full`)
Pin via `CUDA_VISIBLE_DEVICES` + host list. Strategies revised 2026-08-15:
`zero3-hpz2` out (slower than flat everywhere), `fsdp-full` in — the first pass
probed the fabric through a config whose step time is mostly *not* the fabric.
18 cells now, not 15: hpz2 needed ≥4 GPUs and skipped the three 2-GPU rows.

**Read 2B on absolute p50 at fixed GPU count, never on the overhead column** —
each placement divides by its own DDP cell, so the denominator moves with the
row. First pass: within-pair showed the worst ZeRO-3 overhead of any placement
(+47.1%) purely because DDP is fastest on NVLink, while its absolute step time
tied with across-pairs.

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
`ddp / zero3 / fsdp-full` × {2048, 4096, 8192, 16384, 32768} tok/GPU/step (vary
micro-batch, fixed 4096 seq len; the 2048 cell drops to seq_len 2048 because
micro_batch must be ≥1, so it is the one point where attention's quadratic term
also moves). Finds where comm stops mattering. **Most durable output** — a
portable cluster characteristic that transfers to un-benchmarked models.

Strategies revised 2026-08-15: `zero3-hpz2` out, `fsdp-full` in. The first pass
found the crossover (result 4 above), but measured it through DeepSpeed's
stage-3 runtime. Whether the same crossover holds for a backend not paying that
cost decides whether the number is a property of the cluster or of DeepSpeed —
which is the whole claim to durability.

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
- ~~**FSDP2 hybrid**: confirm accelerate 1.14 honors `fsdp_shard_size` for
  `fsdp_version: 2`~~ — ANSWERED on-cluster 2026-08-15: it does not. The key is
  accepted, the 2-D mesh is not built, and the cell ran FULL_SHARD while
  labelled HYBRID_SHARD. Caught by peak memory (12.0 GB, identical to
  fsdp-full's, where a node-local replica must cost GB) — *not* by anything in
  the config or the logs. Row dropped from 2A; `accel_config.py` no longer
  emits the key. **The lesson generalizes: verify a locality feature engaged by
  peak memory moving, not by the config containing the flag.** hpZ passes that
  test (+3.2/+1.6 GB) and is trustworthy as a negative result; fsdp-hybrid
  never did.
- **`zero0` is a stage-0 label over a possibly ZeRO-1-shaped runtime** — see
  row 2 above. The one thing that must not happen is quietly adopting it as the
  denominator without checking; `report.check_baseline_candidates` runs the
  check automatically on every report.
- ~~**DeepSpeed AutoTP** for row 9 — verify 0.19.2 supports it for training~~ —
  ANSWERED on-cluster 2026-08-14: 0.19.2 supports autotp for training at ZeRO
  stage ≤ 2 only, asserting on stage 3. Row 9 is now `tp2-zero2`; `ds_config`
  refuses autotp+stage3 on node0 rather than letting every rank assert. Three
  consequences of TP that are now handled in code and are worth remembering,
  because none of them announces itself:
  - **The HF sampler is not TP-aware.** It shards across all 8 ranks, so the two
    ranks of a TP group get different data and DeepSpeed's one-shot
    "Data inconsistency within the TP group" hook fires on the first forward.
    `train._TPBatchBroadcaster` broadcasts the source rank's batch over the TP
    group (in `_prepare_inputs`, after the batch is on the GPU).
  - **The token control needs scaling on both sides.** A TP group shares one
    batch, so the per-device batch is doubled (`device_micro_batch`) to keep
    8192 tok/GPU/step of compute and the same global batch as DDP, and
    `metrics` divides the per-rank tally by `tp_size` so the cell still groups
    with its DDP baseline. `spec.tokens_per_gpu_step` stays 8192.
  - **The loss gate does not apply to it.** Different sample order and a
    duplicated-batch `num_items_in_batch` both shift the curve; `report`
    reports the row as not gated rather than as failing.
  - Still unverified on hardware: whether HF/DeepSpeed's train_batch_size
    bookkeeping (HF computes it from the full world, DeepSpeed's dp_world_size
    is world/2) is consistent, and whether AutoTP's module coverage on Qwen3
    leaves anything unsharded.
- **`wrapped` packing on a prompt-completion dataset**: confirm trl 1.8's
  `pack_dataset(strategy="wrapped")` carries `completion_mask` through the
  concat-and-chunk and yields chunks of exactly `max_length`. The first run's
  `tokens_per_step_control_held` answers this directly — if it is false, the
  whole real-data switch is invalid and `bfd` is not the fallback (see above);
  the fallback is `--dataset synthetic`.
- **TRL pretokenized path** (synthetic escape hatch only): passes `input_ids` +
  pre-masked `labels` with `skip_prepare_dataset: True`.
- **`_BenchSFTTrainer._prepare_inputs` hook**: token counting moved off
  `training_step` (2026-08-14) so the TP broadcast can run on GPU tensors and
  the count can follow it. Confirm trl 1.8's SFTTrainer does not override
  `_prepare_inputs` in a way that skips the super() call.
- **Qwen3-4B must be staged at `/opt/gpudata/models/Qwen/Qwen3-4B` on both nodes** —
  the new default path, not yet confirmed present. `modeling.check_dense` and
  `modeling.check_full_attention` abort the run if the model at that path is MoE or
  carries linear-attention layers; `describe()` records `layer_type_counts` per run so
  `report.py` has the evidence too.
- `wandb`, `openai` dropped from requirements (unused by the benchmark).
  `flash-attn` was dropped too, then **restored** when timing moved to real
  packed data — see the data section above. It lives in
  `requirements-flash.txt`, not `requirements.txt`.
- **flash-attn build on the cluster**: PyPI ships an sdist, so `pip install`
  compiles kernels. Find the prebuilt wheel matching torch 2.9.1 / cu130 /
  cp312 / cxx11abi from the GitHub releases, confirm it imports on both nodes,
  then pin that exact version in `requirements-flash.txt` — the floor pin there
  is a placeholder.
