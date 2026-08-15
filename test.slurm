#!/bin/bash
#SBATCH -o %J.out        
#SBATCH -e %J.err           
#SBATCH -p gpu_v100     
#SBATCH -J WSVR_test    
#SBATCH --nodes=1          
#SBATCH --gres=gpu:1

module load anaconda3
source activate WSVR
python3 tools/test.py  --experiments experiments/veri776.yml  