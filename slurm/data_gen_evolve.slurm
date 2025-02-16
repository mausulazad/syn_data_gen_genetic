#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 10:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=100G
#SBATCH -N 1 
#SBATCH --output=log/slurm/with_evol_data_gen/log-%A-%a.log 
#SBATCH -J with_evol

export TORCH_HOME=/scratch/mi8uu/mrm/.cache
export HF_DATASETS_CACHE=/scratch/mi8uu/mrm/.cache
export TRANSFORMERS_CACHE=/scratch/mi8uu/cache
export HF_TOKEN=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

module purge
module load apptainer

# python3 main.py --dataset aokvqa --generator_models llama_32 --jury_models llava_critic qwen2_vl

apptainer exec --nv /scratch/mi8uu/mausul/sifs/llava_container_data_gen_v3.sif \
python3 main.py --dataset aokvqa --generator_models llama_32 llava_next molmo --jury_models llava_critic qwen2_vl