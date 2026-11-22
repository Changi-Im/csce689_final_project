#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=40G
#SBATCH --output=./run_sogclr_adamw_%x.%j
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
cd ./myFolder/03.MachineLearning/Final_project
source ./MLProject/bin/activate

export PYTHONPATH="$PYTHONPATH:./iSogCLR/bimodal_exps"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

data_path=./datasets
ann_path=./clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
gamma=0.8
epochs=30
ita_type=sogclr
opt=adamW

CUDA_VISIBLE_DEVICES=0 python ./iSogCLR/bimodal_exps/clip.py \
    --data_path ${data_path} \
    --ann_path ${ann_path} \
    --train_file ${train_file} \
    --train_image_root ${train_image_root} \
    --output_dir output/${opt}/${ita_type}_${data}_g${gamma}_e${epochs} \
    --init_model \
    --use_amp \
    --ita_type ${ita_type} \
    --opt ${opt} \
    --tau_init 0.01 \
    --sogclr_gamma ${gamma} \
    --eta_init 0.03 --sched cosine \
    --no-distributed \
    --epochs ${epochs} \
    --evaluate --evaluate_all --checkpoint ./output/${opt}/${ita_type}_cc3m_g${gamma}_e${epochs}/checkpoint_${epochs}.pth \
    --zs_dataset imagenet --zs_datafolder ./datasets/imagenet/val
