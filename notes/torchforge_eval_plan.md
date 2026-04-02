# TorchForge Evaluation Plan (SuperRL-style)

## Goal
Collect system performance results for TorchForge in the style of SuperRL Fig 10/11:
- Fig 10 analog: trainability (max model size that runs without OOM/hang)
- Fig 11 analog: end-to-end throughput (normalized vs baseline)

## Hardware (Delta)
- Cluster: NCSA Delta
- Allocation: bcjw-delta-gpu
- Target GPUs: start with 1x A100/A40; scale to 4x if available

## Metrics
### Trainability (Fig 10 analog)
- Definition: largest model size that completes at least N training iterations without OOM/hang
- Failure modes: CUDA OOM, host OOM, deadlock/hang
- Record: model name/params, algorithm, GPU count, max batch/seq, reason of failure

### Throughput (Fig 11 analog)
- Primary metric: tokens/sec or steps/sec (whatever TorchForge logs reliably)
- Secondary: wall time per iteration, GPU util, peak HBM
- Normalization: throughput / baseline_throughput for same model+settings

## Baseline
- Baseline candidate: TRL (or TorchForge “no-offload / default” mode if TRL not comparable on Delta)
- Rationale: match SuperRL comparison as closely as practical

## Experiments
1) Sanity run (1 GPU, tiny model, short run) → validate env + logging
2) Throughput sweep (fixed model) → measure tokens/sec vs baseline
3) Trainability sweep (increase model size) → find max stable size
4) Scale-out (4 GPUs) → repeat #2 and #3 if allocation/queue allows

## Logging/Artifacts
- Save stdout logs for each run
- Save sysinfo: nvidia-smi, module list, torch version, git commit hash
- Produce summary tables + 2 plots matching Fig 10/11 style
