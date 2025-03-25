#!/bin/bash

# export CONDA_OVERRIDE_CUDA=11.8
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH


source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate langsplat
export CC=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/gcc-8.5.0
export CXX=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/g++-8.5.0

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/


# first run the autoencoder on each one scene

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/autoencoder
# python train.py --dataset_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/27dd4da69e/dslr --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007  --dataset_name 27dd4da69e

# python test.py --dataset_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/27dd4da69e/dslr  --dataset_name 27dd4da69e 

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/

export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1
python train_ply.py -s /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/27dd4da69e/ -m output/27dd4da69e --start_ply /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannetpp_v1_fix_gs/27dd4da69e/ckpts/point_cloud_30000.ply --feature_level 3


# python preprocess.py --dataset_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/data/lerf_ovs/ramen

# train the autoencoder


# python train.py -s dataset_path -m output/${casename} --start_checkpoint $dataset_path/output/$casename/chkpnt30000.pth --feature_level ${level}


# get the 3-dims language feature of the scene
# python test.py --dataset_name $dataset_path --output