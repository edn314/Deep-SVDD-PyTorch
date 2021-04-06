#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -p fas_gpu
#SBATCH -n 8 # number of cores
#SBATCH --mem 10000 # memory pool for all cores
#SBATCH --gres=gpu:1 # memory pool for all cores
#SBATCH -t 1-00:00 # time (D-HH:MM)

#SBATCH -o _train_%j.%N.out # STDOUT
#SBATCH -e _train_%j.%N.err # STDERR

module load python/3.6.3-fasrc01
module load cuda/9.2.88-fasrc01
source activate deep-svdd
module load GCCcore/6.4.0
nvidia-smi -L

# Run
#python main.py cycif cycif_Net ../log/cycif-run1 /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 16 --ae_weight_decay 0.5e-3;
#python main.py cycif cycif_Net ../log/cycif-run2 /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-3 --pretrain True --ae_lr 0.0001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 16 --ae_weight_decay 0.5e-3;
#python main.py cycif cycif_Net ../log/cycif-run3 /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.001 --n_epochs 200 --lr_milestone 50 --batch_size 16 --weight_decay 5e-3 --pretrain True --ae_lr 0.001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 16 --ae_weight_decay 5e-3;
#python main.py cycif cycif_Net ../log/cycif-run4-no-bias-terms /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 16 --ae_weight_decay 0.5e-3;
#python main.py cycif cycif_Net ../log/cycif-run5-no-pre-training /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --pretrain False
#python main.py cycif cycif_Net ../log/cycif-run6-multi-center /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --pretrain False
#python main.py cycif cycif_Net ../log/cycif-run7-multi-center-k-2-update-with-pretrain /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 16 --ae_weight_decay 0.5e-3;
#python main.py cycif cycif_Net ../log/cycif-run8-multi-center-k-4-update-with-pretrain /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 16 --ae_weight_decay 0.5e-3;
#python main.py cycif cycif_Net ../log/cycif-run9-partial-fit-multi-center-k-4-update-with-pretrain /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 16 --ae_weight_decay 0.5e-3;

# Try loading AE to avoid pretraining
#python main.py cycif cycif_Net ../log/cycif-run-test1 /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 16 --weight_decay 0.5e-6 --load_model /n/pfister_lab2/Lab/enovikov/unsup-ano-detection/anomaly-project/Deep-SVDD-PyTorch/log/cycif-run4-no-bias-terms/model.tar --pretrain False 


# Test
#python main.py cycif cycif_Net ../log/cycif-run-test1 /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --load_config /n/pfister_lab2/Lab/enovikov/unsup-ano-detection/anomaly-project/Deep-SVDD-PyTorch/log/cycif-run5-no-pre-training/config.json --load_model /n/pfister_lab2/Lab/enovikov/unsup-ano-detection/anomaly-project/Deep-SVDD-PyTorch/log/cycif-run5-no-pre-training/model.tar --objective one-class --batch_size 1 --pretrain False
