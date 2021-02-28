#!/bin/bash

#SBATCH -p gpgpu 
#SBATCH --exclusive
#SBATCH -N 8
#SBATCH --job-name=imagent_pytorch    # job name
#SBATCH --ntasks=16           # number of MP tasks
#SBATCH --ntasks-per-node=2       # number of MPI tasks per node
#SBATCH --gres=gpu:2             # number of GPUs per node
#SBATCH --cpus-per-task=1          # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=144:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=imagenet_SGD.out # output file name
#SBATCH --error=imagenet_SGD.err  # error file name

set -x
cd ${SLURM_SUBMIT_DIR}

export NCCL_P2P_DISABLE=1
export NCCL_LL_THRESHOLD=0
export NCCL_SOCKET_IFNAME=ib
export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=mlx5_0


srun python ./imagenet.py --backend=nccl

