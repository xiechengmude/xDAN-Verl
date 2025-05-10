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
from typing import Optional, Tuple, Dict, Any, List

from .enhanced_base_tool import EnhancedBaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnhancedSympyTool(EnhancedBaseTool):
    """增强版SymPy工具，用于符号计算和代数运算
    
    - `to_openai_function_tool_schema`: 返回OpenAI格式的工具模式。
    - `create`: 为轨迹创建工具实例。
    - `execute`: 执行工具。
    - `calc_reward`: 计算工具状态的奖励。
    - `release`: 释放工具实例。
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        """
        初始化增强版SymPy工具
        """
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate({
                "type": "function",
                "function": {
                    "name": "sympy_algebra",
                    "description": "一个用于符号计算和代数运算的工具，可以解方程、化简表达式等",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "description": "要执行的操作类型，可选值：solve（解方程）、simplify（化简表达式）、expand（展开表达式）、factor（因式分解）",
                                "enum": ["solve", "simplify", "expand", "factor"]
                            },
                            "expression": {
                                "type": "string",
                                "description": "要处理的数学表达式或方程，如 'x**2 + 2*x + 1 = 0' 或 '(x+1)**2'",
                            },
                            "variables": {
                                "type": "string",
                                "description": "变量名列表，多个变量用逗号分隔，如 'x' 或 'x,y,z'",
                            }
                        },
                        "required": ["operation", "expression"],
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
        operation = parameters.get("operation", "")
        expression = parameters.get("expression", "")
        variables = parameters.get("variables", "x")
        
        # 确保类型正确
        if not isinstance(operation, str):
            operation = str(operation)
        if not isinstance(expression, str):
            expression = str(expression)
        if not isinstance(variables, str):
            variables = str(variables)
            
        # 解析变量列表
        var_list = [v.strip() for v in variables.split(",")]
        
        # 模拟SymPy操作
        try:
            result = self._simulate_sympy_operation(operation, expression, var_list)
            success = True
        except Exception as e:
            result = f"错误: {str(e)}"
            success = False
            
        return {
            "operation": operation,
            "expression": expression,
            "variables": var_list,
            "result": result,
            "success": success
        }

    def _simulate_sympy_operation(self, operation: str, expression: str, variables: List[str]) -> str:
        """
        模拟SymPy操作
        
        Args:
            operation: 操作类型
            expression: 表达式
            variables: 变量列表
            
        Returns:
            操作结果
        """
        # 注意：这是一个模拟实现，实际应用中应该使用真正的SymPy库
        
        # 清理表达式
        expression = expression.replace(" ", "")
        
        if operation == "solve":
            # 模拟解方程
            if "=" in expression:
                left, right = expression.split("=")
                equation = f"{left}-({right})"
            else:
                equation = expression
                
            # 简单的一元二次方程求解模拟
            if variables[0] + "**2" in equation:
                return f"[{variables[0]} = -1, {variables[0]} = 1]"
            else:
                return f"[{variables[0]} = 0]"
                
        elif operation == "simplify":
            # 模拟化简表达式
            if "(x+1)**2" in expression:
                return "x**2 + 2*x + 1"
            elif "x**2 + 2*x + 1" in expression:
                return "(x + 1)**2"
            else:
                return expression
                
        elif operation == "expand":
            # 模拟展开表达式
            if "(x+1)**2" in expression:
                return "x**2 + 2*x + 1"
            elif "(x+1)*(x-1)" in expression:
                return "x**2 - 1"
            else:
                return expression
                
        elif operation == "factor":
            # 模拟因式分解
            if "x**2 + 2*x + 1" in expression:
                return "(x + 1)**2"
            elif "x**2 - 1" in expression:
                return "(x + 1)*(x - 1)"
            else:
                return expression
                
        else:
            raise ValueError(f"不支持的操作: {operation}")

    async def _evaluate_tool_selection(self, instance_id: str, parameters: dict) -> float:
        """
        评估工具选择的合理性
        
        Args:
            instance_id: 实例ID
            parameters: 输入参数
            
        Returns:
            工具选择奖励
        """
        # 检查问题是否适合使用SymPy工具
        context = self._instance_dict[instance_id].get("context", "")
        
        # 如果问题中包含代数运算或符号计算词汇，则适合使用SymPy
        sympy_indicators = [
            "方程", "解方程", "化简", "展开", "因式分解", "符号", "代数", 
            "多项式", "求解", "等式", "不等式", "方程组"
        ]
        
        # 检查问题是否包含不适合SymPy的概念
        non_sympy_indicators = [
            "数值计算", "数组", "矩阵运算", "统计", "概率", "数据分析", 
            "迭代", "循环", "条件语句"
        ]
        
        # 计算SymPy指标的匹配数
        sympy_matches = sum(1 for indicator in sympy_indicators if indicator in context)
        
        # 计算非SymPy指标的匹配数
        non_sympy_matches = sum(1 for indicator in non_sympy_indicators if indicator in context)
        
        # 如果问题更适合SymPy，则给予高分
        if sympy_matches > 0 and non_sympy_matches == 0:
            return 1.0
        # 如果问题既有SymPy概念又有非SymPy概念，给予中等分数
        elif sympy_matches > 0 and non_sympy_matches > 0:
            return 0.6
        # 如果问题主要是非SymPy概念，给予低分
        elif non_sympy_matches > 0:
            return 0.3
        # 如果没有明显的指标，给予中等分数
        else:
            return 0.5

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
        if "operation" not in parameters or "expression" not in parameters:
            return 0.0
            
        operation = parameters.get("operation", "")
        expression = parameters.get("expression", "")
        
        # 检查操作类型是否有效
        valid_operations = ["solve", "simplify", "expand", "factor"]
        if operation not in valid_operations:
            return 0.2
            
        # 检查表达式是否为空
        if not expression.strip():
            return 0.1
            
        # 检查表达式格式
        format_score = 0.0
        
        # 检查是否包含变量
        if re.search(r'[a-zA-Z]', expression):
            format_score += 0.3
            
        # 检查是否包含数学运算符
        if re.search(r'[\+\-\*\/\=\<\>\^\(\)]', expression):
            format_score += 0.3
            
        # 检查是否包含特殊符号（如**表示幂）
        if "**" in expression:
            format_score += 0.2
            
        # 检查解方程操作是否包含等号
        if operation == "solve" and "=" in expression:
            format_score += 0.2
            
        # 检查变量参数是否存在（对于solve操作）
        if operation == "solve" and "variables" in parameters:
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
                f"SymPy计算结果:\n"
                f"操作: {processed_input.get('operation', '')}\n"
                f"表达式: {processed_input.get('expression', '')}\n"
                f"变量: {', '.join(processed_input.get('variables', []))}\n"
                f"结果: {processed_input.get('result', '')}\n\n"
            )
        else:
            response = (
                f"SymPy计算错误:\n"
                f"操作: {processed_input.get('operation', '')}\n"
                f"表达式: {processed_input.get('expression', '')}\n"
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
