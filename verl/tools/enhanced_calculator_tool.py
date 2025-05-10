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

from .enhanced_base_tool import EnhancedBaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnhancedCalculatorTool(EnhancedBaseTool):
    """增强版计算器工具，用于执行基本的四则运算
    
    - `to_openai_function_tool_schema`: 返回OpenAI格式的工具模式。
    - `create`: 为轨迹创建工具实例。
    - `execute`: 执行工具。
    - `calc_reward`: 计算工具状态的奖励。
    - `release`: 释放工具实例。
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        """
        初始化增强版计算器工具
        """
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate({
                "type": "function",
                "function": {
                    "name": "basic_calculator",
                    "description": "一个用于执行基本四则运算的计算器工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "要计算的数学表达式，支持加减乘除和括号，如 '3 + 4 * (2 - 1)'",
                            }
                        },
                        "required": ["expression"],
                    },
                }
            })
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _process_input(self, instance_id: str, parameters: dict) -> Dict[str, Any]:
        """
        处理输入参数
        
        Args:
            instance_id: 实例ID
            parameters: 输入参数
            
        Returns:
            处理后的参数
        """
        expression = parameters.get("expression", "")
        
        if not isinstance(expression, str):
            expression = str(expression)
            
        # 清理表达式，移除不安全字符
        cleaned_expression = self._clean_expression(expression)
        
        # 计算结果
        try:
            # 使用安全的eval方法计算表达式
            result = self._safe_eval(cleaned_expression)
            success = True
        except Exception as e:
            result = f"错误: {str(e)}"
            success = False
            
        return {
            "raw_expression": expression,
            "cleaned_expression": cleaned_expression,
            "result": result,
            "success": success
        }

    def _clean_expression(self, expression: str) -> str:
        """
        清理表达式，只保留安全的数学运算符和数字
        
        Args:
            expression: 原始表达式
            
        Returns:
            清理后的表达式
        """
        # 移除所有空白字符
        expression = re.sub(r'\s+', '', expression)
        
        # 只保留数字、基本运算符和括号
        expression = re.sub(r'[^0-9\+\-\*\/\(\)\.\,]', '', expression)
        
        return expression

    def _safe_eval(self, expression: str) -> float:
        """
        安全地计算表达式的值
        
        Args:
            expression: 要计算的表达式
            
        Returns:
            计算结果
        """
        # 使用安全的方式计算表达式
        # 注意：这里使用了eval，但在实际生产环境中应该使用更安全的方法
        # 比如使用第三方库如sympy或自己实现表达式解析器
        
        # 创建一个安全的局部命名空间
        safe_dict = {"__builtins__": None}
        
        # 计算表达式
        return eval(expression, {"__builtins__": {}}, safe_dict)

    async def _evaluate_tool_selection(self, instance_id: str, parameters: dict) -> float:
        """
        评估工具选择的合理性
        
        Args:
            instance_id: 实例ID
            parameters: 输入参数
            
        Returns:
            工具选择奖励
        """
        # 检查问题是否适合使用计算器工具
        context = self._instance_dict[instance_id].get("context", "")
        
        # 如果问题中包含简单的数学表达式或运算词汇，则适合使用计算器
        simple_math_indicators = [
            "+", "-", "*", "/", "加", "减", "乘", "除", "等于", "计算", 
            "求和", "总共", "差", "乘积", "商"
        ]
        
        # 检查问题是否包含复杂数学概念，如果有则不太适合使用简单计算器
        complex_math_indicators = [
            "方程", "函数", "微分", "积分", "矩阵", "向量", "概率", 
            "统计", "线性代数", "数组", "列表", "迭代"
        ]
        
        # 计算简单数学指标的匹配数
        simple_matches = sum(1 for indicator in simple_math_indicators if indicator in context)
        
        # 计算复杂数学指标的匹配数
        complex_matches = sum(1 for indicator in complex_math_indicators if indicator in context)
        
        # 如果问题更适合简单计算器，则给予高分
        if simple_matches > 0 and complex_matches == 0:
            return 1.0
        # 如果问题既有简单计算又有复杂概念，给予中等分数
        elif simple_matches > 0 and complex_matches > 0:
            return 0.5
        # 如果问题主要是复杂数学概念，给予低分
        elif complex_matches > 0:
            return 0.2
        # 如果没有明显的数学指标，给予中等分数
        else:
            return 0.7

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
        if "expression" not in parameters:
            return 0.0
            
        expression = parameters.get("expression", "")
        
        # 检查表达式是否为空
        if not expression.strip():
            return 0.1
            
        # 检查表达式格式
        format_score = 0.0
        
        # 检查是否包含数字
        if re.search(r'\d', expression):
            format_score += 0.3
            
        # 检查是否包含运算符
        if re.search(r'[\+\-\*\/]', expression):
            format_score += 0.3
            
        # 检查括号是否匹配
        if expression.count('(') == expression.count(')'):
            format_score += 0.2
            
        # 检查表达式是否可能有效（简单启发式检查）
        if re.match(r'^[\d\+\-\*\/\(\)\s\.]+$', expression):
            format_score += 0.2
            
        return format_score

    async def _evaluate_interpretation(self, instance_id: str, processed_input: dict) -> float:
        """
        评估对工具输出的解释
        
        Args:
            instance_id: 实例ID
            processed_input: 处理后的输入
            
        Returns:
            解释评估奖励
        """
        # 这里简单返回成功与否的分数
        return 1.0 if processed_input.get("success", False) else 0.0

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        计算答案正确性奖励
        
        Args:
            instance_id: 实例ID
            
        Returns:
            正确性奖励
        """
        processed_input = self._instance_dict[instance_id].get("processed_input", {})
        
        # 如果计算成功，给予满分
        if processed_input.get("success", False):
            return 1.0
        else:
            return 0.0

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
        
        # 构建详细反馈
        if processed_input.get("success", False):
            response = (
                f"计算结果:\n"
                f"表达式: {processed_input.get('raw_expression', '')}\n"
                f"结果: {processed_input.get('result', '')}\n\n"
            )
        else:
            response = (
                f"计算错误:\n"
                f"表达式: {processed_input.get('raw_expression', '')}\n"
                f"错误: {processed_input.get('result', '')}\n\n"
            )
            
        response += (
            f"工具评估结果:\n"
            f"- 工具选择评分: {feedback['tool_selection_score']:.2f}\n"
            f"- 参数格式评分: {feedback['parameter_score']:.2f}\n"
            f"- 结果解释评分: {feedback['interpretation_score']:.2f}\n"
            f"- 计算正确性评分: {feedback['correctness_score']:.2f}\n"
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
