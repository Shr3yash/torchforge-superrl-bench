# Stage 1: Get 1-GPU GRPO Running on Delta

## Current Status: Ready to Test

## Files Created
- [x] `run_smoke_1gpu_clean.slurm` - Primary sbatch script with diagnostics
- [x] `run_smoke_1gpu_minimal.slurm` - Fallback script with discovery mode
- [x] `configs/qwen3_1_7b_smoke_nowandb.yaml` - Minimal GRPO config

## Key Fixes Applied
1. **Monarch Timeouts** - Set `MONARCH_STARTUP_TIMEOUT_S=600` (was 30s default)
2. **Single-task topology** - `--ntasks=1` to avoid distributed confusion
3. **Cache redirection** - All HF/Torch caches to `$SLURM_TMPDIR`
4. **Diagnostic logging** - Full env dump before run
5. **Explicit distributed vars** - `WORLD_SIZE=1, RANK=0, LOCAL_RANK=0`

## How to Run
```bash
# On Delta login node:
cd /path/to/torchforge-superrl-bench
sbatch run_smoke_1gpu_clean.slurm
```

## What to Grep For (Success Indicators)
```bash
# Check logs in logs/grpo_smoke_1gpu_<jobid>.out
grep -E "step|loss|throughput|tokens/s|samples/s" logs/*.out
grep "training" logs/*.out
```

## If Still Stuck: Quick Checks
1. Check if stuck at HF download: `grep -i "download\|Downloading" logs/*.out`
2. Check Monarch errors: `grep -i "monarch\|actor\|timeout" logs/*.out`
3. Check config load: `grep -i "config\|yaml\|error" logs/*.err`

---

## Next Steps (After Stage 1 Success)
- [ ] Stage 2: Measure actual throughput
- [ ] Stage 3: Try 4-GPU run
- [ ] Stage 4: Compare with baseline (TRL)
