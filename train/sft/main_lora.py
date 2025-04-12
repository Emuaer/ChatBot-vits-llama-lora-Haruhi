import os
import subprocess

# 设置输出模型目录
output_model = "../../save_folder"
if not os.path.exists(output_model):
    os.makedirs(output_model)

# 设置环境变量
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.2"
os.environ["NCCL_P2P_DISABLE"] = "1"

# 复制 shell 脚本到输出目录（可选）
subprocess.run(["cp", "./finetune.sh", output_model])

# 构建 DeepSpeed 命令
deepspeed_cmd = [
    "deepspeed", "--include", "localhost:0", "finetune_clm_lora_Alpca.py",
    "--model_name_or_path", "/home/chwu/MODELS/Llama3-Chinese-8B-Instruct",
    "--train_files", "/home/chwu/MODELS/Chinese-LLaMA-Alpaca-3-main/Chinese-LLaMA-Alpaca-3-main/data/alpaca_haruhi.json",
    "--validation_files", "/home/chwu/MODELS/Chinese-LLaMA-Alpaca-3-main/Chinese-LLaMA-Alpaca-3-main/valdata/val_harihi.json",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--do_train",
    "--do_eval",
    "--use_fast_tokenizer", "false",
    "--output_dir", output_model,
    "--evaluation_strategy", "steps",
    "--max_eval_samples", "800",
    "--learning_rate", "1e-4",
    "--gradient_accumulation_steps", "8",
    "--num_train_epochs", "30",
    "--warmup_steps", "400",
    "--load_in_bits", "4",
    "--lora_r", "8",
    "--lora_alpha", "48",
    "--target_modules", "q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj",
    "--logging_dir", os.path.join(output_model, "logs"),
    "--logging_strategy", "steps",
    "--logging_steps", "10",
    "--save_strategy", "steps",
    "--preprocessing_num_workers", "10",
    "--save_steps", "1000",
    "--eval_steps", "100",
    "--save_total_limit", "2000",
    "--seed", "42",
    "--disable_tqdm", "false",
    "--ddp_find_unused_parameters", "false",
    "--block_size", "2048",
    "--report_to", "tensorboard",
    "--overwrite_output_dir",
    "--deepspeed", "ds_config_zero2.json",
    "--ignore_data_skip", "true",
    "--bf16",
    "--gradient_checkpointing",
    "--bf16_full_eval",
    "--ddp_timeout", "18000000",
    # "--resume_from_checkpoint", f"{output_model}/checkpoint-20400",  # 可取消注释用于恢复训练
]

# 创建训练日志文件路径
log_path = os.path.join(output_model, "train.log")

# 执行训练命令并实时输出日志
with open(log_path, "a") as log_file:
    process = subprocess.Popen(deepspeed_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
        log_file.write(line)
    process.wait()
