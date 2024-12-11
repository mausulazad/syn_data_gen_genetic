#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 00:35:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=256G
#SBATCH -N 1 
#SBATCH --output=log/slurm/no_evol_data_gen/log-%A-%a.log 
#SBATCH -J no_evol_data_gen_smol

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache
export HF_TOKEN=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

module purge
module load apptainer


apptainer exec --nv /scratch/mi8uu/mausul/sifs/syn_data_gen_evolve_v2.sif \
python3 main.py --dataset aokvqa --generator_models llama_32 llava_next molmo --judge_model phi_3_vision --br_model phi_3_vision