#!/bin/bash
#SBATCH -n 1
#SBATCH -p hpib
#SBATCH -A changsheng
srun -p hpib -n 1 -A changsheng /home/zfwang/project/anaconda3/envs/py27/bin/python transfer.py
