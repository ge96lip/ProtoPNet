#!/bin/bash

# Initialize Conda for the current session
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ProtoPNet
echo "Active environment: $(conda info --envs | grep '*' | awk '{print $1}')"


export PYTHONPATH=./:$PYTHONPATH
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"
# Activate the Conda environment


# Arguments
model="$1"
data_set="$2"
output_dir="$3"
output_dir="./ProtoPFormer/${output_dir}/"
use_gauss="$4"
modeldir="$5"
modeldir="./ProtoPFormer/${modeldir}/"
echo $modeldir
modelfile="$6"

data_path="datasets"
dim=192
batch_size=60

if [ "$model" == "deit_small_patch16_224" ] || [ "$model" == "deit_tiny_patch16_224" ]; then
    reserve_layer_idx=11
elif [ "$model" == "cait_xxs24_224" ]; then
    reserve_layer_idx=1
fi

case "$data_set" in
    "Med")
        global_proto_per_class=3
        prototype_num=600
        reserve_token_nums=81
        ;;
    "ULT")
        global_proto_per_class=3
        prototype_num=30
        reserve_token_nums=81
        ;;
    "PathMNIST")
        global_proto_per_class=3
        prototype_num=45
        reserve_token_nums=81
        ;;
    "ChestMNIST")
        global_proto_per_class=3
        prototype_num=10
        reserve_token_nums=81
        ;;
    "DermaMNIST")
        global_proto_per_class=3
        prototype_num=70
        reserve_token_nums=81
        ;;
    "OCTMNIST")
        global_proto_per_class=3
        prototype_num=20
        reserve_token_nums=81
        ;;
    "PneumoniaMNIST")
        global_proto_per_class=3
        prototype_num=10
        reserve_token_nums=81
        ;;
    "RetinaMNIST")
        global_proto_per_class=3
        prototype_num=25
        reserve_token_nums=81
        ;;
    "BreastMNIST")
        global_proto_per_class=3
        prototype_num=16
        reserve_token_nums=81
        ;;
    "BloodMNIST")
        global_proto_per_class=3
        prototype_num=80
        reserve_token_nums=81
        ;;
    "TissueMNIST")
        global_proto_per_class=3
        prototype_num=40
        reserve_token_nums=81
        ;;
    "OrganAMNIST")
        global_proto_per_class=3
        prototype_num=55
        reserve_token_nums=81
        ;;
    "OrganSMNIST")
        global_proto_per_class=3
        prototype_num=55
        reserve_token_nums=81
        ;;
    "OrganCMNIST")
        global_proto_per_class=3
        prototype_num=55
        reserve_token_nums=81
        ;;
esac

python ./ProtoPFormer/main_visualize_v1.py \
    --finetune=visualize \
    --modeldir="$modeldir" \
    --model="$modelfile" \
    --data_set="$data_set" \
    --data_path="$data_path" \
    --prototype_shape "$prototype_num" "$dim" 1 1 \
    --reserve_layers="$reserve_layer_idx" \
    --reserve_token_nums="$reserve_token_nums" \
    --use_global=True \
    --use_ppc_loss=True \
    --global_coe=0.5 \
    --global_proto_per_class="$global_proto_per_class" \
    --base_architecture="$model" \
    --batch_size="$batch_size" \
    --visual_type=slim_gaussian \
    --output_dir="$output_dir" \
    --use_gauss="$use_gauss" \
    --vis_classes 0 1
