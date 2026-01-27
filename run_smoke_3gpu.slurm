#!/bin/bash
#SBATCH --job-name=grpo_smoke_3gpu
#SBATCH --account=bcjw-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=3
#SBATCH --mem=128g
#SBATCH --time=00:30:00
#SBATCH --output=logs/grpo_smoke_3gpu_%j.out
#SBATCH --error=logs/grpo_smoke_3gpu_%j.err

set -euo pipefail

echo "========================================"
echo "==== GRPO 3-GPU SMOKE TEST - START ===="
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# ============================================
# 1. SLURM ENVIRONMENT DIAGNOSTICS
# ============================================
echo "==== SLURM ENV ===="
env | grep -E "^SLURM_" | sort
echo ""

echo "==== CUDA / GPU ENV ===="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo ""

# ============================================
# 2. SET UP CACHE AND RUNTIME DIRECTORIES (CRITICAL)
# ============================================
# Use scratch space for caches to avoid HOME quota issues
export SCRATCH_CACHE="${SLURM_TMPDIR:-/tmp/slurm_${SLURM_JOB_ID}}"
mkdir -p "$SCRATCH_CACHE"

# CRITICAL: Monarch/hyperactor uses XDG_RUNTIME_DIR for temp files
# On compute nodes, /run/user/$UID often doesn't exist - redirect it
export XDG_RUNTIME_DIR="$SCRATCH_CACHE/runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Also set TMPDIR explicitly for any other temp file needs
export TMPDIR="$SCRATCH_CACHE/tmp"
mkdir -p "$TMPDIR"

export HF_HOME="$SCRATCH_CACHE/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$SCRATCH_CACHE/torch"
export TRITON_CACHE_DIR="$SCRATCH_CACHE/triton"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

echo "==== CACHE PATHS ===="
echo "SCRATCH_CACHE=$SCRATCH_CACHE"
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
echo "TMPDIR=$TMPDIR"
echo "HF_HOME=$HF_HOME"
echo "TORCH_HOME=$TORCH_HOME"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
# Test write access
touch "$HF_HOME/.write_test" && rm "$HF_HOME/.write_test" && echo "Cache write test: OK"
touch "$XDG_RUNTIME_DIR/.write_test" && rm "$XDG_RUNTIME_DIR/.write_test" && echo "XDG_RUNTIME_DIR write test: OK"
echo ""

# ============================================
# 3. MONARCH / HYPERACTOR TIMEOUTS (CRITICAL)
# ============================================
# These prevent stalls during actor mesh initialization
export HYPERACTOR_CODEC_MAX_FRAME_LENGTH=134217728
export MONARCH_STARTUP_TIMEOUT_S=600
export MONARCH_RPC_TIMEOUT_S=600
export MONARCH_ACTOR_RESPAWN_TIMEOUT_S=300

# Additional hyperactor settings
export HYPERACTOR_TIMEOUT=600
export HYPERACTOR_SPAWN_TIMEOUT=600
export HYPERACTOR_MESH_TIMEOUT_SECS=600

# Enable Rust backtrace for better debugging
export RUST_BACKTRACE=1

echo "==== MONARCH ENV ===="
echo "HYPERACTOR_CODEC_MAX_FRAME_LENGTH=$HYPERACTOR_CODEC_MAX_FRAME_LENGTH"
echo "MONARCH_STARTUP_TIMEOUT_S=$MONARCH_STARTUP_TIMEOUT_S"
echo "MONARCH_RPC_TIMEOUT_S=$MONARCH_RPC_TIMEOUT_S"
echo "HYPERACTOR_TIMEOUT=$HYPERACTOR_TIMEOUT"
echo "HYPERACTOR_SPAWN_TIMEOUT=$HYPERACTOR_SPAWN_TIMEOUT"
echo "RUST_BACKTRACE=$RUST_BACKTRACE"
echo ""

# ============================================
# 4. DISTRIBUTED / NCCL CONFIG
# ============================================
# For single-GPU, minimize distributed overhead
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Single-process mode - avoid unnecessary distributed init
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "==== DISTRIBUTED ENV ===="
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo ""

# ============================================
# 5. PYTHON ENVIRONMENT
# ============================================
export PYTHON_BIN="/u/sbhatkar/micromamba/envs/forge/bin/python"
export TORCHFORGE_ROOT="/u/sbhatkar/work/torchforge"
export PYTHONPATH="$TORCHFORGE_ROOT:${PYTHONPATH:-}"

echo "==== PYTHON ENV ===="
echo "PYTHON_BIN=$PYTHON_BIN"
echo "TORCHFORGE_ROOT=$TORCHFORGE_ROOT"
echo "PYTHONPATH=$PYTHONPATH"
$PYTHON_BIN --version
echo ""

# ============================================
# 6. PREFLIGHT CHECKS
# ============================================
echo "==== PREFLIGHT: IMPORT TESTS ===="
$PYTHON_BIN -c "
import sys
print(f'Python: {sys.executable}')

print('Testing torch...', end=' ')
import torch
print(f'OK (version={torch.__version__}, cuda={torch.cuda.is_available()})')

print('Testing vllm...', end=' ')
import vllm
print('OK')

print('Testing transformers...', end=' ')
import transformers
print(f'OK (version={transformers.__version__})')

print('Testing torchforge apps.grpo.main...', end=' ')
import apps.grpo.main
print('OK')

print('All imports successful!')
" || { echo "PREFLIGHT FAILED: Import error"; exit 1; }
echo ""

# ============================================
# 7. CONFIGURATION
# ============================================
# CRITICAL: Use SLURM_SUBMIT_DIR, not $(dirname "$0") which fails in batch mode
REPO_ROOT="${SLURM_SUBMIT_DIR:-/u/sbhatkar/work/torchforge-superrl-bench}"
CONFIG_FILE="${1:-configs/qwen3_1_7b_smoke_nowandb.yaml}"
FULL_CONFIG_PATH="$REPO_ROOT/$CONFIG_FILE"

echo "==== CONFIG ===="
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "REPO_ROOT=$REPO_ROOT"
echo "Config file: $FULL_CONFIG_PATH"

if [[ ! -f "$FULL_CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found at $FULL_CONFIG_PATH"
    echo "Available configs in repo:"
    ls -la "$REPO_ROOT/configs/" 2>/dev/null || echo "No configs dir"
    exit 1
fi

echo "Config found! Contents:"
cat "$FULL_CONFIG_PATH"
echo ""

# ============================================
# 8. RUN GRPO
# ============================================
echo "========================================"
echo "==== RUN ===="
echo "========================================"
echo "Start time: $(date)"
echo ""

set -x  # Print exact command being executed

cd "$TORCHFORGE_ROOT"

$PYTHON_BIN -m apps.grpo.main \
    --config "$FULL_CONFIG_PATH" \
    2>&1 | tee "$REPO_ROOT/logs/grpo_3gpu_run_${SLURM_JOB_ID}.log"

set +x

echo ""
echo "========================================"
echo "==== RUN COMPLETE ===="
echo "========================================"
echo "End time: $(date)"
echo "Exit code: $?"
