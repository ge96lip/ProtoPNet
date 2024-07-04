@echo off
setlocal

:: Ensure that the Conda environment is activated
call activate i2DL2

:: Set the environment variables
set "PYTHONPATH=.\;%PYTHONPATH%"
set "CUDA_VISIBLE_DEVICES=0"

:: Arguments
set "model=%1"
set "batch_size=%2"

:: Set hyperparameters and training settings
set "seed=1028"
set "warmup_lr=1e-4"
set "warmup_epochs=5"
set "features_lr=1e-4"
set "add_on_layers_lr=3e-3"
set "prototype_vectors_lr=3e-3"
set "opt=adamw"
set "sched=cosine"
set "decay_epochs=10"
set "decay_rate=0.1"
set "weight_decay=0.05"
set "epochs=200"
set "output_dir=output_cosine/"
set "input_size=224"
set "use_global=True"
set "use_ppc_loss=True"
set "last_reserve_num=81"
set "global_coe=0.5"
set "ppc_cov_thresh=1."
set "ppc_mean_thresh=2."
set "global_proto_per_class=10"
set "ppc_cov_coe=0.1"
set "ppc_mean_coe=0.5"
set "dim=192"

:: Determine specific model settings
if "%model%" == "deit_small_patch16_224" (
    set "reserve_layer_idx=11"
) else if "%model%" == "deit_tiny_patch16_224" (
    set "reserve_layer_idx=11"
) else if "%model%" == "cait_xxs24_224" (
    set "reserve_layer_idx=1"
)

set "ft=protopformer"
set "data_set=Med"
set "prototype_num=600"
set "data_path=datasets"

:: Run the Python script-
python main.py ^
    --base_architecture=%model% ^
    --data_set=%data_set% ^
    --data_path=%data_path% ^
    --input_size=%input_size% ^
    --output_dir=%output_dir%/%data_set%/%model%/%seed%-%opt%-%weight_decay%-%epochs%-%ft% ^
    --model=%model% ^
    --batch_size=%batch_size% ^
    --seed=%seed% ^
    --opt=%opt% ^
    --sched=%sched% ^
    --warmup-epochs=%warmup_epochs% ^
    --warmup-lr=%warmup_lr% ^
    --decay-epochs=%decay_epochs% ^
    --decay-rate=%decay_rate% ^
    --weight_decay=%weight_decay% ^
    --epochs=%epochs% ^
    --finetune=%ft% ^
    --features_lr=%features_lr% ^
    --add_on_layers_lr=%add_on_layers_lr% ^
    --prototype_vectors_lr=%prototype_vectors_lr% ^
    --prototype_shape %prototype_num% %dim% 1 1 ^
    --reserve_layers %reserve_layer_idx% ^
    --reserve_token_nums %last_reserve_num% ^
    --use_global=%use_global% ^
    --use_ppc_loss=%use_ppc_loss% ^
    --ppc_cov_thresh=%ppc_cov_thresh% ^
    --ppc_mean_thresh=%ppc_mean_thresh% ^
    --global_coe=%global_coe% ^
    --global_proto_per_class=%global_proto_per_class% ^
    --ppc_cov_coe=%ppc_cov_coe% ^
    --ppc_mean_coe=%ppc_mean_coe%

endlocal
