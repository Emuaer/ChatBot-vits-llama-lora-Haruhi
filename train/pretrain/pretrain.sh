output_model=../../output_model
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./pretrain.sh ${output_model}
cp ./ds_config_zero*.json ${output_model}
export CUDA_HOME=/usr/local/cuda/
export NCCL_P2P_DISABLE=1

deepspeed --include localhost:0 pretrain_clm.py \
    --config_name  /home/chwu/MODELS/Llama3-Chinese-8B-Instruct/config.json \
    --tokenizer_name /home/chwu/MODELS/Llama3-Chinese-8B-Instruct \
    --train_files ../../data/wiki_zh/train_hari_v2.csv \
    --validation_files  ../../data/wiki_zh/dev_hari_v2.csv \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --use_fast_tokenizer false \
    --max_eval_samples 500 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --warmup_steps 5000 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 500 \
    --eval_steps 5000000 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --overwrite_output_dir \
    --report_to tensorboard \
    --run_name ${output_model} \
    --bf16 \
    --bf16_full_eval \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero3.json \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
