#!/bin/bash
# GRU search (Step 5) on Northeastern Explorer, CPU node.
#
# Resumes from reports/step5_gru_search.partial.json if present, so if the job hits its
# time limit, just `sbatch slurm/train_gru.sbatch` again and it continues from the last
# finished config. CPU on purpose: the first configs of this search ran on CPU, and the
# rest should run on the same kind of hardware.
#
# Needs (not in git): data/processed/seq_a.npz, data/processed/features_a.parquet.
# Hospital B files are NOT needed and should not be copied - the test hospital stays sealed.
#
# Submit from the repo root:   sbatch slurm/train_gru.sbatch
#SBATCH --job-name=sepsis-gru
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=reports/slurm-gru-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3            # if this fails: `module avail anaconda` and use the exact name
eval "$(conda shell.bash hook)"
conda activate sepsis

echo "node $(hostname), $SLURM_CPUS_PER_TASK cpus, started $(date)"
python -u src/train_gru.py --threads "$SLURM_CPUS_PER_TASK"
echo "finished $(date)"
