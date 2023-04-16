
TOT_CUDA="0,1,2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="1337"

DATA_PATH="./data/ckb_clean.json"
EPOCHS=3
MICRO_BATCH_SIZE=28
HUB_MODEL_ID="razhan/kurdish-llama"
HUB_TOKEN=""

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune.py \
--data_path $DATA_PATH \
--micro_batch_size $MICRO_BATCH_SIZE \
--epochs $EPOCHS \
--push_to_hub true \
--hub_model_id $HUB_MODEL_ID\
--hub_token $HUB_TOKEN \
--wandb true