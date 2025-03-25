#!/bin/bash
#SBATCH --job-name=nnunet_baseline
#SBATCH --output=sbatch_log/convformer0_2layer_acdc_inception_conv_elu_debug_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 120GB


# install 
conda create -n langsplat_cu18 python=3.9 -y

conda activate langsplat_cu18

export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 cuda-nvcc=11.8 -y 
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install numpy==1.24.1 tqdm opencv-python scikit-image scikit-learn matplotlib tensorboardX plyfile colorama



conda create -n langsplat_cu18 python=3.9 -y

conda activate langsplat_cu18

export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

# conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 cuda-nvcc=11.8 -y 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install numpy==1.24.1 tqdm opencv-python scikit-image scikit-learn matplotlib tensorboardX plyfile 