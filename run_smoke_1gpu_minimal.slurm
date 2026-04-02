#!/bin/bash
#SBATCH --job-name=grpo_min_1gpu
#SBATCH --account=bcjw-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=logs/grpo_minimal_1gpu_%j.out
#SBATCH --error=logs/grpo_minimal_1gpu_%j.err

set -euo pipefail

echo "========================================"
echo "==== MINIMAL GRPO TEST - START ===="
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# ============================================
# SLURM ENVIRONMENT
# ============================================
echo "==== SLURM ENV ===="
env | grep -E "^SLURM_" | sort || true
echo ""
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
echo ""

# ============================================
# CACHE DIRECTORIES
# ============================================
export SCRATCH_CACHE="${SLURM_TMPDIR:-/tmp/slurm_${SLURM_JOB_ID}}"
mkdir -p "$SCRATCH_CACHE"

export HF_HOME="$SCRATCH_CACHE/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$SCRATCH_CACHE/torch"
export TRITON_CACHE_DIR="$SCRATCH_CACHE/triton"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

echo "HF_HOME=$HF_HOME"
echo ""

# ============================================
# MONARCH TIMEOUTS (CRITICAL)
# ============================================
export HYPERACTOR_CODEC_MAX_FRAME_LENGTH=134217728
export MONARCH_STARTUP_TIMEOUT_S=600
export MONARCH_RPC_TIMEOUT_S=600
export MONARCH_ACTOR_RESPAWN_TIMEOUT_S=300

echo "MONARCH_STARTUP_TIMEOUT_S=$MONARCH_STARTUP_TIMEOUT_S"
echo ""

# ============================================
# DISTRIBUTED CONFIG (SINGLE-PROCESS)
# ============================================
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# ============================================
# PYTHON ENVIRONMENT
# ============================================
export PYTHON_BIN="/u/sbhatkar/micromamba/envs/forge/bin/python"
export TORCHFORGE_ROOT="/u/sbhatkar/work/torchforge"
export PYTHONPATH="$TORCHFORGE_ROOT:${PYTHONPATH:-}"

echo "==== PYTHON CHECK ===="
$PYTHON_BIN --version
$PYTHON_BIN -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

# ============================================
# DISCOVERY: Find torchforge entry points
# ============================================
echo "==== TORCHFORGE DISCOVERY ===="
echo "Looking for entry points in $TORCHFORGE_ROOT..."

if [[ -f "$TORCHFORGE_ROOT/apps/grpo/main.py" ]]; then
    echo "Found: apps/grpo/main.py"
    head -50 "$TORCHFORGE_ROOT/apps/grpo/main.py" 2>/dev/null || true
fi

echo ""
echo "Looking for example configs..."
find "$TORCHFORGE_ROOT" -name "*.yaml" -o -name "*.yml" 2>/dev/null | head -10 || true
echo ""

# Check if there's a CLI help
echo "==== CLI HELP (if available) ===="
$PYTHON_BIN -m apps.grpo.main --help 2>&1 | head -50 || echo "No --help available"
echo ""

# ============================================
# ATTEMPT RUN
# ============================================
echo "========================================"
echo "==== RUN ===="
echo "========================================"
echo "Start time: $(date)"
echo ""

# CRITICAL: Use SLURM_SUBMIT_DIR, not $(dirname "$0") which fails in batch mode
REPO_ROOT="${SLURM_SUBMIT_DIR:-/u/sbhatkar/work/torchforge-superrl-bench}"
CONFIG_FILE="$REPO_ROOT/configs/qwen3_1_7b_smoke_nowandb.yaml"

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "REPO_ROOT=$REPO_ROOT"
echo "CONFIG_FILE=$CONFIG_FILE"

cd "$TORCHFORGE_ROOT"

set -x

# Try with config file
if [[ -f "$CONFIG_FILE" ]]; then
    $PYTHON_BIN -m apps.grpo.main --config "$CONFIG_FILE" 2>&1 || {
        echo ""
        echo "Config-based run failed, trying with minimal args..."
        # Fallback: try without config
        $PYTHON_BIN -m apps.grpo.main 2>&1 || true
    }
else
    echo "No config file found at $CONFIG_FILE"
    $PYTHON_BIN -m apps.grpo.main 2>&1 || true
fi

set +x

echo ""
echo "========================================"
echo "==== DONE ===="
echo "========================================"
echo "End time: $(date)"
