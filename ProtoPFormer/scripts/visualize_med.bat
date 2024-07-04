@echo off
setlocal

:: Arguments
set "model=%1"
set "data_set=%2"
set "output_dir=%3"
set "use_gauss=%4"
set "modeldir=%5"
set "modelfile=%6"

set "data_path=datasets"
set "dim=192"
set "batch_size=60"

if "%model%"=="deit_small_patch16_224" (
    set "reserve_layer_idx=11"
) else if "%model%"=="deit_tiny_patch16_224" (
    set "reserve_layer_idx=11"
) else if "%model%"=="cait_xxs24_224" (
    set "reserve_layer_idx=1"
)

if "%data_set%"=="CUB2011U" (
    set "global_proto_per_class=10"
    set "prototype_num=2000"
    set "reserve_token_nums=81"
) else if "%data_set%"=="Car" (
    set "global_proto_per_class=5"
    set "prototype_num=1960"
    set "reserve_token_nums=121"
) else if "%data_set%"=="Dogs" (
    set "global_proto_per_class=5"
    set "prototype_num=1200"
    set "reserve_token_nums=81"
) else if "%data_set%"=="Med" (
    set "global_proto_per_class=5"
    set "prototype_num=600"
    set "reserve_token_nums=81"
)


python main_visualize_v1.py ^
    --finetune=visualize ^
    --modeldir=%modeldir% ^
    --model=%modelfile% ^
    --data_set=%data_set% ^
    --data_path=%data_path% ^
    --prototype_shape %prototype_num% %dim% 1 1 ^
    --reserve_layers=%reserve_layer_idx% ^
    --reserve_token_nums=%reserve_token_nums% ^
    --use_global=True ^
    --use_ppc_loss=True ^
    --global_coe=0.5 ^
    --global_proto_per_class=%global_proto_per_class% ^
    --base_architecture=%model% ^
    --batch_size=%batch_size% ^
    --visual_type=slim_gaussian ^
    --output_dir=%output_dir% ^
    --use_gauss=%use_gauss% ^
    --vis_classes 0 1 2

endlocal
