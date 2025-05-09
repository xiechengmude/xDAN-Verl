from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4
import json

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

class EnhancedBaseTool(BaseTool):
    """增强版基础工具类，提供精细化奖励计算功能"""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # 配置奖励权重
        self.reward_weights = {
            "tool_selection": config.get("tool_selection_weight", 0.2),
            "parameter": config.get("parameter_weight", 0.3),
            "interpretation": config.get("interpretation_weight", 0.2),
            "correctness": config.get("correctness_weight", 0.5),
        }
        
        # 配置动态奖励参数
        self.dynamic_reward = config.get("dynamic_reward", False)
        self.early_phase = config.get("early_phase", 30)
        self.middle_phase = config.get("middle_phase", 100)
        self.initial_scale = config.get("initial_scale", 0.5)
        self.mid_scale = config.get("mid_scale", 0.8)
        self.full_scale = config.get("full_scale", 1.0)

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, 
                     context: Optional[str] = None, expected_params: Optional[dict] = None,
                     difficulty: float = 1.0, training_step: int = 0, **kwargs) -> str:
        """创建工具实例，增加了更多上下文信息"""
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "context": context or "",
            "expected_params": expected_params or {},
            "difficulty": difficulty,
            "training_step": training_step,
            "reward_history": [],
            "total_reward": 0.0,
            "call_count": 0,
        }
        
        return instance_id

    async def execute(self, instance_id: str, parameters: str, **kwargs) -> Tuple[str, float, dict]:
        """执行工具调用，计算精细化奖励"""
        # 增加调用次数
        self._instance_dict[instance_id]["call_count"] += 1
        
        # 解析参数
        try:
            _parameters = json.loads(parameters)
        except json.JSONDecodeError:
            _parameters = {}
            
        # 处理输入参数
        processed_input = self._process_input(instance_id, _parameters)
        
        # 计算各维度奖励
        tool_selection_reward = await self._evaluate_tool_selection(instance_id, _parameters)
        parameter_reward = await self._evaluate_parameters(instance_id, _parameters)
        
        # 计算答案正确性奖励
        correctness_reward = await self.calc_reward(instance_id)
        
        # 计算结果解释奖励 (默认为0，由子类实现)
        interpretation_reward = await self._evaluate_interpretation(instance_id, processed_input)
        
        # 计算加权总奖励
        current_reward = (
            self.reward_weights["tool_selection"] * tool_selection_reward +
            self.reward_weights["parameter"] * parameter_reward +
            self.reward_weights["interpretation"] * interpretation_reward +
            self.reward_weights["correctness"] * correctness_reward
        )
        
        # 应用动态奖励调整
        if self.dynamic_reward:
            current_reward *= self._get_dynamic_reward_scale(instance_id)
        
        # 计算工具调用增量奖励
        previous_reward = self._instance_dict[instance_id]["total_reward"]
        reward_improvement = current_reward - previous_reward
        
        # 确定工具调用奖励
        if reward_improvement > 0:
            # 有改进，给予正向奖励
            tool_reward = min(0.2, reward_improvement)  # 最高0.2的正向奖励
        elif reward_improvement == 0:
            # 无改进，给予中性奖励
            tool_reward = 0.0
        else:
            # 性能下降，给予负向奖励
            tool_reward = max(-0.1, reward_improvement)  # 最低-0.1的负向奖励
            
        # 更新奖励历史和总奖励
        self._instance_dict[instance_id]["reward_history"].append(current_reward)
        self._instance_dict[instance_id]["total_reward"] = current_reward
        
        # 构建详细反馈
        feedback = {
            "tool_selection_score": tool_selection_reward,
            "parameter_score": parameter_reward,
            "interpretation_score": interpretation_reward,
            "correctness_score": correctness_reward,
            "total_score": current_reward,
            "improvement": reward_improvement,
        }
        
        return self._format_response(instance_id, processed_input, feedback), tool_reward, feedback

    async def _evaluate_tool_selection(self, instance_id: str, parameters: dict) -> float:
        """评估工具选择的合理性，默认实现返回1.0，子类可重写"""
        return 1.0

    async def _evaluate_parameters(self, instance_id: str, parameters: dict) -> float:
        """评估参数的完整性和正确性"""
        try:
            expected_params = self._instance_dict[instance_id].get("expected_params", {})
            
            if not expected_params:
                return 1.0  # 如果没有预期参数，默认满分
                
            # 参数名称匹配
            param_keys = set(parameters.keys())
            expected_keys = set(expected_params.keys())
            
            if not expected_keys:
                return 1.0
                
            name_match = len(param_keys & expected_keys) / len(expected_keys | param_keys) if expected_keys | param_keys else 1.0
            
            # 参数值匹配
            value_matches = 0
            for k in expected_keys:
                if k in param_keys and parameters[k] == expected_params[k]:
                    value_matches += 1
                    
            value_match = value_matches / len(expected_keys) if expected_keys else 1.0
            
            # 综合分数 (名称匹配占40%，值匹配占60%)
            return 0.4 * name_match + 0.6 * value_match
        except Exception:
            return 0.0

    async def _evaluate_interpretation(self, instance_id: str, processed_input: any) -> float:
        """评估对工具输出的解释，默认实现返回0.0，子类可重写"""
        return 0.0

    def _get_dynamic_reward_scale(self, instance_id: str) -> float:
        """根据任务难度和训练进度动态调整奖励比例"""
        difficulty = self._instance_dict[instance_id].get("difficulty", 1.0)
        training_step = self._instance_dict[instance_id].get("training_step", 0)
        
        if training_step < self.early_phase:
            # 训练初期，给予更高奖励以鼓励探索
            base_scale = self.initial_scale
        elif training_step < self.middle_phase:
            # 训练中期，逐渐提高标准
            progress = (training_step - self.early_phase) / (self.middle_phase - self.early_phase)
            base_scale = self.initial_scale + progress * (self.mid_scale - self.initial_scale)
        else:
            # 训练后期，使用完整奖励
            base_scale = self.full_scale
            
        # 根据任务难度调整
        # 简单任务给予更高初始奖励，困难任务随训练进度增加奖励
        difficulty_factor = 1.0 + (difficulty - 1.0) * min(1.0, training_step / self.middle_phase)
        
        return base_scale * difficulty_factor

    def _process_input(self, instance_id: str, parameters: dict) -> any:
        """处理输入参数，默认实现直接返回参数，子类可重写"""
        return parameters

    def _format_response(self, instance_id: str, processed_input: any, feedback: dict) -> str:
        """格式化响应，默认实现返回简单反馈，子类可重写"""
        return (f"Tool call feedback:\n"
                f"- Selection score: {feedback['tool_selection_score']:.2f}\n"
                f"- Parameter score: {feedback['parameter_score']:.2f}\n"
                f"- Correctness score: {feedback['correctness_score']:.2f}\n"
                f"- Total score: {feedback['total_score']:.2f}")
