#!/bin/bash

# 设置训练参数
export PYTHONPATH=/data/vayu/train/xDAN-Verl
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 运行增强版训练脚本
python -m verl.train.train_rl \
    --config-path=examples/sglang_multiturn/configs \
    --config-name=qwen2.5-3b_gsm8k_multiturn_4xgpu \
    multi_turn.tool_config_path=examples/sglang_multiturn/configs/enhanced_gsm8k_tool.yaml \
    rollout_class=verl.workers.rollout.sglang_rollout.enhanced_async_sglang_rollout.EnhancedAsyncSGLangRollout \
    rollout_request_class=verl.workers.rollout.enhanced_schemas.EnhancedAsyncRolloutRequest \
    output_dir=outputs/enhanced_gsm8k_multiturn \
    wandb.name=enhanced_gsm8k_multiturn
