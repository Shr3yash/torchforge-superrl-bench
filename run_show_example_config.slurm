#!/bin/bash
#SBATCH --job-name=show_config
#SBATCH --account=bcjw-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=8g
#SBATCH --time=00:05:00
#SBATCH --output=logs/show_config_%j.out
#SBATCH --error=logs/show_config_%j.err

# Quick job to show torchforge example configs (no GPU needed)

TORCHFORGE_ROOT="/u/sbhatkar/work/torchforge"

echo "==== TORCHFORGE EXAMPLE CONFIGS ===="
echo ""

echo "=== apps/grpo/qwen3_1_7b.yaml ==="
cat "$TORCHFORGE_ROOT/apps/grpo/qwen3_1_7b.yaml" 2>/dev/null || echo "NOT FOUND"
echo ""

echo "=== Looking for other GRPO configs ==="
find "$TORCHFORGE_ROOT/apps/grpo" -name "*.yaml" -o -name "*.yml" 2>/dev/null | while read f; do
    echo "--- $f ---"
    cat "$f"
    echo ""
done

echo "=== Apps directory structure ==="
ls -la "$TORCHFORGE_ROOT/apps/grpo/" 2>/dev/null || true
