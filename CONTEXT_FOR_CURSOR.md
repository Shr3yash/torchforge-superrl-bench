# torchforge-superrl-bench — Ground Truth Context

## 1. Objective (Non-Negotiable)
Reproduce *minimal* GRPO system-level performance results using PyTorch Forge (torchforge),
in the spirit of SuperRL Figures 10 and 11, on NCSA Delta GPUs.

This is NOT about full paper reproduction.
This IS about:
- verifying torchforge GRPO runs correctly on Delta
- producing stable throughput numbers
- enabling scale-up experiments (1 GPU → multi-GPU)

Target deliverable: working GRPO runs + logged throughput/step-time metrics.

Deadline pressure: hours, not days.

---

## 2. Hardware / Cluster Reality
Cluster: NCSA Delta  
GPU tested so far: NVIDIA A40 (46 GB)  
Scheduler: Slurm  
Account: bcjw-delta-gpu  

Login node: login.delta.ncsa.illinois.edu  
Compute nodes: gpubXXX.delta.ncsa.illinois.edu  

Important constraints:
- HOME quota is small (~50GB)
- HuggingFace cache must be redirected
- Monarch actor startup is slow unless timeouts are increased
- Jobs MUST be launched via sbatch (not interactive Python)

---

## 3. Repo Layout (Authoritative)
```
torchforge-superrl-bench/
├── configs/
│   ├── qwen3_1_7b_smoke.yaml
│   └── qwen3_1_7b_smoke_nowandb.yaml  # wandb removed
├── run_smoke_1gpu_clean.slurm         # single GPU smoke test
├── run_grpo_3gpu.slurm                # multi-GPU attempt
├── logs/
├── results/
├── scripts/
└── README.md
```

TorchForge source lives at:
```
/u/sbhatkar/work/torchforge
```

This repo is ONLY a benchmark harness.

---

## 4. Python / Environment
Python: 3.12  
Env path:
```
/u/sbhatkar/micromamba/envs/forge/bin/python
```

torchforge is NOT pip-installed.
It is imported via:
```
export PYTHONPATH=/u/sbhatkar/work/torchforge
```

Import proof works:
```bash
python -c "import apps.grpo.main"
```

---

## 5. Current Failure Mode (Critical)
Symptoms:
- Slurm job allocates GPU successfully
- torchforge GRPO starts
- No progress after "==== RUN ===="
- Actor meshes stall or timeout
- Multi-GPU jobs sit in PENDING (Resources) for long periods

Observed error earlier:
```
Failed to initialize replica 0: error spawning actor mesh
Timeout(30s)
```

Likely causes:
- Monarch actor mesh startup timeout too low
- Incorrect Slurm task / GPU / process topology
- Missing Slurm environment variables expected by torchforge
- NCCL / CUDA_VISIBLE_DEVICES misalignment
- Too many logical actors for a single node config

---

## 6. What *Already* Works
- SSH access
- Slurm allocation
- CUDA visible on compute node
- torchforge imports
- vLLM detects CUDA
- YAML config loads
- wandb removed cleanly
- HF cache redirected
- A40 GPU visible via nvidia-smi

---

## 7. What Must Be Done (In Order)
1. Make **1-GPU GRPO run actually execute steps**
2. Log:
   - step time
   - samples/sec OR tokens/sec (whatever torchforge exposes)
3. Only then attempt multi-GPU
4. Do NOT jump to GH200 / multi-node yet

---

## 8. Hard Constraints for Any Fix
- No wandb
- No interactive debugging
- No rewriting torchforge internals unless unavoidable
- Prefer Slurm/env fixes over code changes
- Favor minimal configs over “paper-accurate” configs

---

## 9. Known Relevant Env Vars (Use These)
```
HYPERACTOR_CODEC_MAX_FRAME_LENGTH=134217728
MONARCH_STARTUP_TIMEOUT_S=300
MONARCH_RPC_TIMEOUT_S=300

HF_HOME=$SLURM_TMPDIR/hf
HF_DATASETS_CACHE=$HF_HOME/datasets
TORCH_HOME=$SLURM_TMPDIR/torch

OMP_NUM_THREADS=1
CUDA_DEVICE_MAX_CONNECTIONS=1
```

---

## 10. Success Criteria
A run is successful if:
- GRPO executes >1 training step
- Logs show progress (loss, rollout, update, or timing)
- Job exits cleanly or can be interrupted manually
- Results are reproducible on repeated submission

Anything else is failure.

---

## 11. Mental Model You Should Use
Treat torchforge like a distributed system, not a script.
Startup correctness > performance.
Once stable, scaling is trivial.

Do NOT optimize early.
