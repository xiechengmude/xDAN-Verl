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
from typing import Optional, Tuple, Dict, Any, List

from .enhanced_base_tool import EnhancedBaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnhancedPythonExecutorTool(EnhancedBaseTool):
    """增强版Python执行工具，用于执行Python代码解决问题
    
    - `to_openai_function_tool_schema`: 返回OpenAI格式的工具模式。
    - `create`: 为轨迹创建工具实例。
    - `execute`: 执行工具。
    - `calc_reward`: 计算工具状态的奖励。
    - `release`: 释放工具实例。
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        """
        初始化增强版Python执行工具
        """
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate({
                "type": "function",
                "function": {
                    "name": "python_executor",
                    "description": "一个用于执行Python代码解决问题的工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "要执行的Python代码，可以包含多行代码块",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "对代码的解释说明",
                            }
                        },
                        "required": ["code"],
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
        code = parameters.get("code", "")
        explanation = parameters.get("explanation", "")
        
        # 确保类型正确
        if not isinstance(code, str):
            code = str(code)
        if not isinstance(explanation, str):
            explanation = str(explanation)
            
        # 执行Python代码
        try:
            stdout, stderr, result = self._safe_execute_python(code)
            success = stderr == ""
            output = stdout if success else stderr
        except Exception as e:
            result = None
            success = False
            output = f"执行错误: {str(e)}"
            
        return {
            "code": code,
            "explanation": explanation,
            "result": result,
            "output": output,
            "success": success
        }

    def _safe_execute_python(self, code: str) -> Tuple[str, str, Any]:
        """
        安全地执行Python代码
        
        Args:
            code: 要执行的Python代码
            
        Returns:
            标准输出、标准错误和执行结果
        """
        # 创建安全的执行环境
        local_vars = {}
        
        # 捕获标准输出和标准错误
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # 执行代码
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # 编译代码
                compiled_code = compile(code, "<string>", "exec")
                
                # 执行代码
                exec(compiled_code, {"__builtins__": __builtins__}, local_vars)
                
                # 检查是否有返回值（最后一个表达式的值）
                result = None
                if "result" in local_vars:
                    result = local_vars["result"]
                
            # 获取输出
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # 限制输出长度
            if len(stdout) > self.max_output_length:
                stdout = stdout[:self.max_output_length] + "... (输出被截断)"
                
            return stdout, stderr, result
            
        except Exception as e:
            return "", str(e), None

    async def _evaluate_tool_selection(self, instance_id: str, parameters: dict) -> float:
        """
        评估工具选择的合理性
        
        Args:
            instance_id: 实例ID
            parameters: 输入参数
            
        Returns:
            工具选择奖励
        """
        # 检查问题是否适合使用Python执行工具
        context = self._instance_dict[instance_id].get("context", "")
        
        # 如果问题中包含编程或算法词汇，则适合使用Python执行工具
        python_indicators = [
            "编程", "算法", "代码", "函数", "循环", "迭代", "条件", "变量", 
            "列表", "字典", "集合", "类", "对象", "模块", "导入", "计算"
        ]
        
        # 检查问题是否包含不适合Python执行的概念
        non_python_indicators = [
            "符号计算", "代数方程", "方程求解"
        ]
        
        # 计算Python指标的匹配数
        python_matches = sum(1 for indicator in python_indicators if indicator in context)
        
        # 计算非Python指标的匹配数
        non_python_matches = sum(1 for indicator in non_python_indicators if indicator in context)
        
        # 如果问题更适合Python执行，则给予高分
        if python_matches > 0 and non_python_matches == 0:
            return 1.0
        # 如果问题既有Python概念又有非Python概念，给予中等分数
        elif python_matches > 0 and non_python_matches > 0:
            return 0.7
        # 如果问题主要是非Python概念，给予低分
        elif non_python_matches > 0:
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
        if "code" not in parameters:
            return 0.0
            
        code = parameters.get("code", "")
        explanation = parameters.get("explanation", "")
        
        # 检查代码是否为空
        if not code.strip():
            return 0.1
            
        # 检查代码格式
        format_score = 0.0
        
        # 检查是否包含基本Python语法
        if re.search(r'def|if|for|while|import|print|return', code):
            format_score += 0.2
            
        # 检查是否有变量赋值
        if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=', code):
            format_score += 0.2
            
        # 检查是否有函数定义
        if re.search(r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', code):
            format_score += 0.2
            
        # 检查缩进是否一致
        if not re.search(r'^\s+\S+.*\n^\S+', code, re.MULTILINE):
            format_score += 0.2
            
        # 检查是否提供了解释
        if explanation.strip():
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
        explanation = processed_input.get("explanation", "")
        
        # 如果没有提供解释，给予低分
        if not explanation.strip():
            return 0.2
            
        # 评估解释的质量
        quality_score = 0.0
        
        # 检查解释的长度
        if len(explanation) > 50:
            quality_score += 0.3
            
        # 检查解释中是否包含关键词
        if re.search(r'函数|变量|循环|条件|计算|结果|输出|返回|算法|步骤', explanation):
            quality_score += 0.3
            
        # 检查解释是否与代码相关
        code = processed_input.get("code", "")
        code_keywords = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code)
        explanation_contains_code_keywords = any(keyword in explanation for keyword in code_keywords if len(keyword) > 2)
        
        if explanation_contains_code_keywords:
            quality_score += 0.4
            
        return quality_score

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        计算答案正确性奖励
        
        Args:
            instance_id: 实例ID
            
        Returns:
            正确性奖励
        """
        processed_input = self._instance_dict[instance_id].get("processed_input", {})
        
        # 如果执行成功，给予满分
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
                f"Python执行结果:\n"
                f"输出:\n```\n{processed_input.get('output', '')}\n```\n"
            )
            
            if processed_input.get("result") is not None:
                response += f"返回值: {processed_input.get('result')}\n\n"
        else:
            response = (
                f"Python执行错误:\n"
                f"错误信息:\n```\n{processed_input.get('output', '')}\n```\n\n"
            )
            
        response += (
            f"工具评估结果:\n"
            f"- 工具选择评分: {feedback['tool_selection_score']:.2f}\n"
            f"- 参数格式评分: {feedback['parameter_score']:.2f}\n"
            f"- 代码解释评分: {feedback['interpretation_score']:.2f}\n"
            f"- 执行正确性评分: {feedback['correctness_score']:.2f}\n"
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
