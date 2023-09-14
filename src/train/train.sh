MODEL=yahma/llama-7b-hf
DATA=./train.json

mkdir -p saved_models
torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path ${MODEL}  \
    --data_path ${DATA} \
    --run_name universalner \
    --bf16 True \
    --output_dir saved_models/universalner \
    --dataloader_num_workers 8 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config ./fsdp_config.json \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True
