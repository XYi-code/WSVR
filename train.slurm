#!/bin/bash
#SBATCH -o %J.out
#SBATCH -e %J.err
#SBATCH -p gpu_v100
#SBATCH -J WSVR_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load anaconda3
source activate WSVR
python3 tools/train.py --experiments experiments/veri776.yml --suffix 'VeRi'