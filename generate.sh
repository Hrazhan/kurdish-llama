BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="razhan/kurdish-llama"
TYPE_WRITER=1 # whether output streamly

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH \
    --use_typewriter $TYPE_WRITER