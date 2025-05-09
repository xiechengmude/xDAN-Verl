# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizer

from verl.workers.rollout.schemas import AsyncRolloutRequest, FinishReasonTypeEnum, Message
from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema


class EnhancedAsyncRolloutRequest(AsyncRolloutRequest):
    """增强版异步回滚请求，支持保存工具反馈信息"""

    # 添加工具反馈字段
    tool_feedbacks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # 添加训练信息字段
    training_step: int = 0
    difficulty: float = 1.0
    
    # 添加奖励分解字段
    reward_components: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
        tool_feedbacks: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        完成请求处理，保存奖励分数和工具反馈
        
        Args:
            tokenizer: 分词器
            reward_scores: 奖励分数
            finish_reason_type: 完成原因类型
            tool_feedbacks: 工具反馈信息
        """
        self.state = self.StateEnum.COMPLETED
        self.reward_scores = reward_scores
        
        # 保存工具反馈信息
        if tool_feedbacks:
            self.tool_feedbacks = tool_feedbacks
            
            # 提取奖励组件
            for tool_name, feedback in tool_feedbacks.items():
                if isinstance(feedback, dict):
                    components = {}
                    for key, value in feedback.items():
                        if key.endswith('_score') and isinstance(value, (int, float)):
                            components[key] = value
                    if components:
                        self.reward_components[tool_name] = components
        
        self.response_ids = self.input_ids[len(self.prompt_ids):]
        
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
            
        self.truncate_output_ids(tokenizer)
        
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""
