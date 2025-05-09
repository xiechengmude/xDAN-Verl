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

import json
import logging
import os
import re
from typing import Optional, Tuple, Dict, Any

from verl.utils.reward_score import gsm8k

from .enhanced_base_tool import EnhancedBaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnhancedGsm8kTool(EnhancedBaseTool):
    """增强版GSM8K工具，提供精细化奖励计算功能

    - `to_openai_function_tool_schema`: 返回OpenAI格式的工具模式。
    - `create`: 为轨迹创建工具实例。
    - `execute`: 执行工具。
    - `calc_reward`: 计算工具状态的奖励。
    - `release`: 释放工具实例。
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        """
        初始化增强版GSM8K工具
        """
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate({
                "type": "function",
                "function": {
                    "name": "calc_gsm8k_reward",
                    "description": "一个用于计算GSM8K问题答案正确性的工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "问题的答案",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "解题推理过程",
                            }
                        },
                        "required": ["answer"],
                    },
                }
            })
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, 
                     context: Optional[str] = None, expected_params: Optional[dict] = None,
                     difficulty: float = 1.0, training_step: int = 0, **kwargs) -> str:
        """
        创建工具实例
        
        Args:
            instance_id: 实例ID
            ground_truth: 正确答案
            context: 问题上下文
            expected_params: 预期参数
            difficulty: 任务难度
            training_step: 训练步骤
            
        Returns:
            实例ID
        """
        return await super().create(
            instance_id=instance_id,
            ground_truth=ground_truth,
            context=context,
            expected_params=expected_params,
            difficulty=difficulty,
            training_step=training_step,
            **kwargs
        )

    def _process_input(self, instance_id: str, parameters: dict) -> Dict[str, Any]:
        """
        处理输入参数
        
        Args:
            instance_id: 实例ID
            parameters: 输入参数
            
        Returns:
            处理后的参数
        """
        answer = parameters.get("answer", "")
        reasoning = parameters.get("reasoning", "")
        
        if not isinstance(answer, str):
            answer = str(answer)
            
        # 格式化答案
        if not answer.startswith("#### "):
            formatted_answer = "#### " + answer
        else:
            formatted_answer = answer
            
        return {
            "raw_answer": answer,
            "formatted_answer": formatted_answer,
            "reasoning": reasoning,
        }

    async def execute(self, instance_id: str, parameters: str, **kwargs) -> Tuple[str, float, dict]:
        """
        执行工具调用
        
        Args:
            instance_id: 实例ID
            parameters: 输入参数
            
        Returns:
            响应、奖励和反馈
        """
        # 保存原始响应
        self._instance_dict[instance_id]["last_raw_response"] = parameters
        
        # 调用父类的execute方法计算精细化奖励
        return await super().execute(instance_id, parameters, **kwargs)

    async def _evaluate_tool_selection(self, instance_id: str, parameters: dict) -> float:
        """
        评估工具选择的合理性
        
        Args:
            instance_id: 实例ID
            parameters: 输入参数
            
        Returns:
            工具选择奖励
        """
        # 对于GSM8K问题，如果问题需要数学计算，则选择此工具是合理的
        # 这里简单返回1.0，实际应用中可根据问题特性判断
        return 1.0

    async def _evaluate_parameters(self, instance_id: str, parameters: dict) -> float:
        """
        评估参数的完整性和正确性
        
        Args:
            instance_id: 实例ID
            parameters: 输入参数
            
        Returns:
            参数评估奖励
        """
        # 检查必要参数是否存在
        if "answer" not in parameters:
            return 0.0
            
        # 检查参数格式
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            return 0.5  # 答案类型不正确，但可转换
            
        # 检查答案格式
        format_score = 0.0
        if answer.strip():
            # 检查是否包含数字
            if re.search(r'\d', answer):
                format_score += 0.5
                
            # 检查是否有格式化（如#### 前缀）
            if answer.startswith("#### "):
                format_score += 0.5
            
        # 检查是否提供了推理过程
        reasoning_score = 0.0
        if "reasoning" in parameters and parameters["reasoning"].strip():
            reasoning_score = 1.0
            
        # 综合评分
        return 0.6 * format_score + 0.4 * reasoning_score

    async def _evaluate_interpretation(self, instance_id: str, processed_input: dict) -> float:
        """
        评估对工具输出的解释
        
        Args:
            instance_id: 实例ID
            processed_input: 处理后的输入
            
        Returns:
            解释评估奖励
        """
        reasoning = processed_input.get("reasoning", "")
        
        if not reasoning:
            return 0.0
            
        # 评估推理过程的质量
        quality_score = 0.0
        
        # 检查推理中是否包含计算步骤
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', reasoning):
            quality_score += 0.5
            
        # 检查推理是否有明确的步骤标记
        if re.search(r'步骤|step|首先|然后|接下来|最后', reasoning, re.IGNORECASE):
            quality_score += 0.3
            
        # 检查推理是否与答案一致
        raw_answer = processed_input.get("raw_answer", "")
        if raw_answer and raw_answer in reasoning:
            quality_score += 0.2
            
        return quality_score

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        计算答案正确性奖励
        
        Args:
            instance_id: 实例ID
            
        Returns:
            正确性奖励
        """
        # 获取格式化后的答案
        formatted_answer = self._instance_dict[instance_id].get("processed_input", {}).get("formatted_answer", "")
        if not formatted_answer and "response" in self._instance_dict[instance_id]:
            # 兼容旧版本
            formatted_answer = self._instance_dict[instance_id]["response"]
            
        # 计算GSM8K奖励
        return gsm8k.compute_score(
            formatted_answer,
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    def _format_response(self, instance_id: str, processed_input: dict, feedback: dict) -> str:
        """
        格式化响应
        
        Args:
            instance_id: 实例ID
            processed_input: 处理后的输入
            feedback: 反馈信息
            
        Returns:
            格式化的响应
        """
        # 保存处理后的输入，供calc_reward使用
        self._instance_dict[instance_id]["processed_input"] = processed_input
        self._instance_dict[instance_id]["response"] = processed_input.get("formatted_answer", "")
        
        # 构建详细反馈
        response = (
            f"GSM8K工具评估结果:\n"
            f"- 答案: {processed_input.get('raw_answer', '')}\n"
            f"- 工具选择评分: {feedback['tool_selection_score']:.2f}\n"
            f"- 参数格式评分: {feedback['parameter_score']:.2f}\n"
            f"- 推理解释评分: {feedback['interpretation_score']:.2f}\n"
            f"- 答案正确性评分: {feedback['correctness_score']:.2f}\n"
            f"- 总评分: {feedback['total_score']:.2f}"
        )
        
        if feedback.get('improvement', 0) > 0:
            response += f"\n- 相比上次提交有 {feedback['improvement']:.2f} 的改进"
        elif feedback.get('improvement', 0) < 0:
            response += f"\n- 相比上次提交下降了 {-feedback['improvement']:.2f}"
            
        return response

    async def release(self, instance_id: str, **kwargs) -> None:
        """
        释放工具实例
        
        Args:
            instance_id: 实例ID
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
