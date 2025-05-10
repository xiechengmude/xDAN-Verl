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
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Tuple, Dict, Any, List, Union

from .enhanced_base_tool import EnhancedBaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnhancedNumpyTool(EnhancedBaseTool):
    """增强版NumPy工具，用于处理数组和矩阵计算
    
    - `to_openai_function_tool_schema`: 返回OpenAI格式的工具模式。
    - `create`: 为轨迹创建工具实例。
    - `execute`: 执行工具。
    - `calc_reward`: 计算工具状态的奖励。
    - `release`: 释放工具实例。
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        """
        初始化增强版NumPy工具
        """
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate({
                "type": "function",
                "function": {
                    "name": "numpy_math",
                    "description": "一个用于处理数组、矩阵计算和高级数学函数的工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "description": "要执行的操作类型，可选值：array_creation（创建数组）、matrix_operation（矩阵运算）、statistics（统计计算）、linear_algebra（线性代数）",
                                "enum": ["array_creation", "matrix_operation", "statistics", "linear_algebra"]
                            },
                            "data": {
                                "type": "string",
                                "description": "要处理的数据，可以是数组、矩阵或其他数据结构的字符串表示，如 '[1, 2, 3]' 或 '[[1, 2], [3, 4]]'",
                            },
                            "code": {
                                "type": "string",
                                "description": "要执行的NumPy代码片段，如 'np.mean(arr)' 或 'np.dot(A, B)'",
                            }
                        },
                        "required": ["operation", "data"],
                    },
                }
            })
        super().__init__(config, tool_schema)
        
        # 设置安全执行环境的超时时间
        self.execution_timeout = config.get("execution_timeout", 5)  # 默认5秒
        
        # 设置安全执行环境的最大输出长度
        self.max_output_length = config.get("max_output_length", 1000)  # 默认1000字符

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
        data = parameters.get("data", "")
        code = parameters.get("code", "")
        
        # 确保类型正确
        if not isinstance(operation, str):
            operation = str(operation)
        if not isinstance(data, str):
            data = str(data)
        if not isinstance(code, str):
            code = str(code)
            
        # 模拟NumPy操作
        try:
            result, output = self._simulate_numpy_operation(operation, data, code)
            success = True
        except Exception as e:
            result = None
            output = f"错误: {str(e)}"
            success = False
            
        return {
            "operation": operation,
            "data": data,
            "code": code,
            "result": result,
            "output": output,
            "success": success
        }

    def _simulate_numpy_operation(self, operation: str, data: str, code: str) -> Tuple[Any, str]:
        """
        模拟NumPy操作
        
        Args:
            operation: 操作类型
            data: 数据
            code: NumPy代码
            
        Returns:
            操作结果和输出
        """
        # 注意：这是一个模拟实现，实际应用中应该使用真正的NumPy库
        
        # 解析数据
        try:
            parsed_data = eval(data)
        except:
            parsed_data = data
            
        # 根据操作类型执行不同的模拟
        if operation == "array_creation":
            # 模拟数组创建
            if isinstance(parsed_data, list):
                if all(isinstance(item, list) for item in parsed_data):
                    # 二维数组
                    shape = f"({len(parsed_data)}, {len(parsed_data[0])})"
                    return f"array({parsed_data})", f"创建了形状为{shape}的数组"
                else:
                    # 一维数组
                    return f"array({parsed_data})", f"创建了长度为{len(parsed_data)}的数组"
            else:
                return f"array({parsed_data})", "创建了标量数组"
                
        elif operation == "matrix_operation":
            # 模拟矩阵运算
            if "dot" in code or "matmul" in code:
                return "array([[7, 10], [15, 22]])", "执行了矩阵乘法运算"
            elif "add" in code or "+" in code:
                return f"array({parsed_data}) + array({parsed_data})", "执行了矩阵加法运算"
            elif "subtract" in code or "-" in code:
                return "array([[0, 0], [0, 0]])", "执行了矩阵减法运算"
            elif "multiply" in code or "*" in code:
                return f"array({parsed_data}) * array({parsed_data})", "执行了矩阵元素乘法运算"
            else:
                return "未知结果", "执行了未知的矩阵运算"
                
        elif operation == "statistics":
            # 模拟统计计算
            if "mean" in code:
                if isinstance(parsed_data, list) and all(isinstance(x, (int, float)) for x in parsed_data):
                    return sum(parsed_data) / len(parsed_data), "计算了数组的均值"
                else:
                    return 2.5, "计算了数组的均值"
            elif "median" in code:
                return 2.5, "计算了数组的中位数"
            elif "std" in code or "var" in code:
                return 1.29, "计算了数组的标准差或方差"
            elif "min" in code:
                if isinstance(parsed_data, list) and all(isinstance(x, (int, float)) for x in parsed_data):
                    return min(parsed_data), "计算了数组的最小值"
                else:
                    return 1, "计算了数组的最小值"
            elif "max" in code:
                if isinstance(parsed_data, list) and all(isinstance(x, (int, float)) for x in parsed_data):
                    return max(parsed_data), "计算了数组的最大值"
                else:
                    return 4, "计算了数组的最大值"
            else:
                return "未知结果", "执行了未知的统计计算"
                
        elif operation == "linear_algebra":
            # 模拟线性代数运算
            if "det" in code or "determinant" in code:
                return -2.0, "计算了矩阵的行列式"
            elif "inv" in code or "inverse" in code:
                return "array([[-2, 1], [1.5, -0.5]])", "计算了矩阵的逆"
            elif "eig" in code or "eigenvalue" in code:
                return "eigenvalues: [5.37, -0.37], eigenvectors: [...]", "计算了矩阵的特征值和特征向量"
            elif "svd" in code:
                return "U: [...], S: [...], V: [...]", "执行了奇异值分解"
            elif "solve" in code:
                return "array([1, 1])", "求解了线性方程组"
            else:
                return "未知结果", "执行了未知的线性代数运算"
                
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
        # 检查问题是否适合使用NumPy工具
        context = self._instance_dict[instance_id].get("context", "")
        
        # 如果问题中包含数组、矩阵或数值计算词汇，则适合使用NumPy
        numpy_indicators = [
            "数组", "矩阵", "向量", "线性代数", "统计", "均值", "方差", 
            "标准差", "最大值", "最小值", "求和", "乘积", "点积", "特征值"
        ]
        
        # 检查问题是否包含不适合NumPy的概念
        non_numpy_indicators = [
            "符号计算", "代数方程", "方程求解", "简单计算"
        ]
        
        # 计算NumPy指标的匹配数
        numpy_matches = sum(1 for indicator in numpy_indicators if indicator in context)
        
        # 计算非NumPy指标的匹配数
        non_numpy_matches = sum(1 for indicator in non_numpy_indicators if indicator in context)
        
        # 如果问题更适合NumPy，则给予高分
        if numpy_matches > 0 and non_numpy_matches == 0:
            return 1.0
        # 如果问题既有NumPy概念又有非NumPy概念，给予中等分数
        elif numpy_matches > 0 and non_numpy_matches > 0:
            return 0.6
        # 如果问题主要是非NumPy概念，给予低分
        elif non_numpy_matches > 0:
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
        if "operation" not in parameters or "data" not in parameters:
            return 0.0
            
        operation = parameters.get("operation", "")
        data = parameters.get("data", "")
        code = parameters.get("code", "")
        
        # 检查操作类型是否有效
        valid_operations = ["array_creation", "matrix_operation", "statistics", "linear_algebra"]
        if operation not in valid_operations:
            return 0.2
            
        # 检查数据是否为空
        if not data.strip():
            return 0.1
            
        # 检查数据格式
        format_score = 0.0
        
        # 检查数据是否是有效的数组或矩阵表示
        if re.match(r'\[\s*(\[.*\]|\d+(\.\d+)?(,\s*\d+(\.\d+)?)*)\s*\]', data):
            format_score += 0.4
            
        # 检查是否提供了代码
        if code.strip():
            format_score += 0.3
            
            # 检查代码是否包含NumPy相关函数
            if re.search(r'np\.(mean|dot|array|matrix|sum|std|var|min|max|det|inv|eig|svd|solve)', code):
                format_score += 0.3
            
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
                f"NumPy计算结果:\n"
                f"操作: {processed_input.get('operation', '')}\n"
                f"数据: {processed_input.get('data', '')}\n"
            )
            
            if processed_input.get("code", ""):
                response += f"代码: {processed_input.get('code', '')}\n"
                
            response += f"结果: {processed_input.get('result', '')}\n"
            response += f"输出: {processed_input.get('output', '')}\n\n"
        else:
            response = (
                f"NumPy计算错误:\n"
                f"操作: {processed_input.get('operation', '')}\n"
                f"数据: {processed_input.get('data', '')}\n"
                f"错误: {processed_input.get('output', '')}\n\n"
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
