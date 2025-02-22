#!/bin/bash -l
#SBATCH -J styelgan2-unet-training
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=13-00:00:00
#SBATCH --partition=gpu-qi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ataffe
#SBATCH -o jobs/train_stylegan-%j.output
#SBATCH -e jobs/train_stylegan-%j.error

# Run the Python script with the input file
module load conda3/4.X
conda activate stylegan3
python train.py --outdir=training-runs --cfg=stylegan2-unet --data=/home/ataffe/SyntheticData/stylegan3/datasets/FLIm-Images-no-phantom-cropped-256x256.zip \
 --gpus=1 --batch=32 --gamma=15 --mirror=1 --snap=30 --map-depth=2 --glr=0.0025 --dlr=0.0025 --seg_mask=1 --cutmix=1 --p=0.3
