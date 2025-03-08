#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0-29
#SBATCH --ntasks=1 
#SBATCH -t 3-00:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --mem=64G
#SBATCH -N 1 
#SBATCH --output=log/slurm/with_evol_data_gen/log-%A-%a.log 
#SBATCH -J datagen

export TORCH_HOME=/scratch/mi8uu/mrm/.cache
export HF_DATASETS_CACHE=/scratch/mi8uu/mrm/.cache
export TRANSFORMERS_CACHE=/scratch/mi8uu/cache
export HF_TOKEN=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

module purge
module load apptainer

# split from 0 to 29
splits=(chunk_0 chunk_1 chunk_2 chunk_3 chunk_4 chunk_5 chunk_6 chunk_7 chunk_8 chunk_9 chunk_10 chunk_11 chunk_12 chunk_13 chunk_14 chunk_15 chunk_16 chunk_17 chunk_18 chunk_19 chunk_20 chunk_21 chunk_22 chunk_23 chunk_24 chunk_25 chunk_26 chunk_27 chunk_28 chunk_29)
split=${splits[$SLURM_ARRAY_TASK_ID]}


# python3 main.py --dataset aokvqa --generator_models llama_32 --jury_models llava_critic qwen2_vl
# python3 main.py --dataset aokvqa --generator_models llama_32 llava_next molmo --jury_models llava_critic qwen2_vl

apptainer exec --nv /scratch/mi8uu/mausul/sifs/llava_container_data_gen_v3.sif \
python3 main.py --dataset aokvqa --dataset_split ${split} --generator_models llama_32 --jury_models llava_critic