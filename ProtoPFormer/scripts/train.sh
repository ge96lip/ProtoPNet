#!/bin/bash

# Ensure that the Conda environment is activated
source activate protop

# Set the environment variables
export PYTHONPATH=.:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

# Arguments
model=$1
batch_size=$2
data_set=$3
prototype_num=$4

# Construct paths
main_py_path="./ProtoPFormer/main.py"

data_path="./ProtoPFormer/datasets"
output_dir="./ProtoPFormer/output_cosine"
echo $output_dir

# Set hyperparameters and training settings
seed=1028
warmup_lr=1e-4
warmup_epochs=5
features_lr=1e-4
add_on_layers_lr=3e-3
prototype_vectors_lr=3e-3
opt=adamw
sched=cosine
decay_epochs=10
decay_rate=0.1
weight_decay=0.05
epochs=100
input_size=224
use_global=True
use_ppc_loss=True
last_reserve_num=81
global_coe=0.5
ppc_cov_thresh=1.
ppc_mean_thresh=2.
global_proto_per_class=3
ppc_cov_coe=0.1
ppc_mean_coe=0.5
dim=192

# Determine specific model settings
if [ "$model" == "deit_small_patch16_224" ] || [ "$model" == "deit_tiny_patch16_224" ]; then
    reserve_layer_idx=11
elif [ "$model" == "cait_xxs24_224" ]; then
    reserve_layer_idx=1
fi

ft=protopformer

# Run the Python script
python "$main_py_path" \
    --base_architecture=$model \
    --data_set=$data_set \
    --data_path=$data_path \
    --input_size=$input_size \
    --output_dir=$output_dir/$data_set/$model/$seed-$opt-$weight_decay-$epochs-$ft \
    --model=$model \
    --batch_size=$batch_size \
    --seed=$seed \
    --opt=$opt \
    --sched=$sched \
    --warmup-epochs=$warmup_epochs \
    --warmup-lr=$warmup_lr \
    --decay-epochs=$decay_epochs \
    --decay-rate=$decay_rate \
    --weight_decay=$weight_decay \
    --epochs=$epochs \
    --finetune=$ft \
    --features_lr=$features_lr \
    --add_on_layers_lr=$add_on_layers_lr \
    --prototype_vectors_lr=$prototype_vectors_lr \
    --prototype_shape $prototype_num $dim 1 1 \
    --reserve_layers $reserve_layer_idx \
    --reserve_token_nums $last_reserve_num \
    --use_global=$use_global \
    --use_ppc_loss=$use_ppc_loss \
    --ppc_cov_thresh=$ppc_cov_thresh \
    --ppc_mean_thresh=$ppc_mean_thresh \
    --global_coe=$global_coe \
    --global_proto_per_class=$global_proto_per_class \
    --ppc_cov_coe=$ppc_cov_coe \
    --ppc_mean_coe=$ppc_mean_coe
