# sglang_multiturn 工具奖励机制优化方案

本文档详细描述了基于 ToolRL 和 AgentTool 奖励设计理念，对 sglang_multiturn 工具奖励机制的优化方案。

## 一、当前奖励机制分析

当前 sglang_multiturn 工具奖励机制主要包含两个部分：

1. **工具调用奖励**：
   ```python
   # 在 Gsm8kTool.execute 方法中
   tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
   ```

2. **结果正确性奖励**：
   ```python
   # 在 Gsm8kTool.calc_reward 方法中
   return gsm8k.compute_score(
       self._instance_dict[instance_id]["response"],
       self._instance_dict[instance_id]["ground_truth"],
       method="flexible",
       format_score=0.0,
       score=1.0,
   )
   ```

这种设计存在以下局限性：
- 奖励信号过于粗粒度，缺乏对工具调用过程的精细评估
- 不区分格式正确性和语义准确性
- 缺乏对中间推理过程的评估
- 没有考虑任务难度和训练进度的动态调整

## 二、优化方案步骤

### 步骤 1：创建通用工具基类

创建一个新的通用工具基类 `EnhancedBaseTool`，作为所有工具类的父类，提供增强的奖励计算功能。

```python
# 文件路径：verl/tools/enhanced_base_tool.py

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
```

### 步骤 2：优化 Gsm8kTool 类

修改 `Gsm8kTool` 类，继承自新的 `EnhancedBaseTool` 类，并实现特定的奖励计算逻辑。

```python
# 文件路径：verl/tools/gsm8k_tool.py

import json
import logging
import os
import re
from typing import Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score import gsm8k

from .enhanced_base_tool import EnhancedBaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Gsm8kTool(EnhancedBaseTool):
    """增强版 GSM8K 工具，提供精细化奖励计算"""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        
        # GSM8K 特定配置
        self.answer_pattern = re.compile(r'(\d+)')
        self.thinking_pattern = re.compile(r'(Let me|I need to|First|To solve|We need to|I\'ll|Let\'s)')

    async def _evaluate_tool_selection(self, instance_id: str, parameters: dict) -> float:
        """评估工具选择的合理性"""
        # 对于 GSM8K 问题，检查上下文是否包含数学问题的特征
        context = self._instance_dict[instance_id].get("context", "")
        
        # 检查上下文是否包含数字和数学运算符号
        has_numbers = bool(re.search(r'\d+', context))
        has_operators = bool(re.search(r'[\+\-\*\/\=]', context))
        
        # 检查是否包含数学问题关键词
        math_keywords = ['calculate', 'sum', 'total', 'difference', 'product', 
                         'divide', 'multiply', 'add', 'subtract', 'how many', 
                         'how much', 'cost', 'price', 'percent', 'average']
        
        has_math_keywords = any(keyword in context.lower() for keyword in math_keywords)
        
        # 计算工具选择分数
        if has_numbers and (has_operators or has_math_keywords):
            return 1.0  # 非常适合使用计算工具
        elif has_numbers or has_math_keywords:
            return 0.8  # 可能适合使用计算工具
        else:
            return 0.5  # 不确定是否适合使用计算工具

    async def _evaluate_parameters(self, instance_id: str, parameters: dict) -> float:
        """评估参数的完整性和正确性"""
        # 首先使用基类的参数评估逻辑
        base_score = await super()._evaluate_parameters(instance_id, parameters)
        
        # GSM8K 特定评估：检查答案格式
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
            
        # 检查答案是否包含数字
        has_number = bool(self.answer_pattern.search(answer))
        
        # 检查答案是否有合理的长度
        reasonable_length = 1 <= len(answer) <= 20
        
        # 计算格式分数
        format_score = 0.0
        if has_number and reasonable_length:
            format_score = 1.0
        elif has_number:
            format_score = 0.8
        elif reasonable_length:
            format_score = 0.3
            
        # 综合评分 (基础参数评分占60%，格式评分占40%)
        return 0.6 * base_score + 0.4 * format_score

    async def _evaluate_interpretation(self, instance_id: str, processed_input: any) -> float:
        """评估对工具输出的解释"""
        # 获取答案
        answer = processed_input.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
            
        # 检查是否包含推理过程
        has_thinking = bool(self.thinking_pattern.search(answer))
        
        # 检查推理过程的长度
        thinking_length = len(answer.split())
        
        # 计算解释分数
        if has_thinking and thinking_length >= 20:
            return 1.0  # 详细的推理过程
        elif has_thinking and thinking_length >= 10:
            return 0.8  # 基本的推理过程
        elif has_thinking:
            return 0.5  # 简短的推理过程
        else:
            return 0.0  # 没有推理过程

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """计算答案正确性奖励"""
        # 使用现有的 GSM8K 评分逻辑
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    def _process_input(self, instance_id: str, parameters: dict) -> dict:
        """处理输入参数"""
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
            
        # 格式化答案
        if answer.startswith("#### "):
            self._instance_dict[instance_id]["response"] = answer
        else:
            self._instance_dict[instance_id]["response"] = "#### " + answer
            
        return parameters

    def _format_response(self, instance_id: str, processed_input: dict, feedback: dict) -> str:
        """格式化响应"""
        answer = processed_input.get("answer", "")
        correctness = feedback["correctness_score"]
        
        # 构建详细反馈
        response = (
            f"Current parsed answer: {answer}\n"
            f"Correctness: {correctness:.2f}\n"
            f"Tool selection score: {feedback['tool_selection_score']:.2f}\n"
            f"Parameter quality: {feedback['parameter_score']:.2f}\n"
            f"Reasoning quality: {feedback['interpretation_score']:.2f}\n"
            f"Total score: {feedback['total_score']:.2f}"
        )
        
        return response
```

### 步骤 3：创建通用工具配置文件

创建一个通用工具配置文件，支持不同类型的问题和工具。

```yaml
# 文件路径：examples/sglang_multiturn/config/tool_config/enhanced_tool_config.yaml

tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    config:
      # 奖励权重配置
      tool_selection_weight: 0.2
      parameter_weight: 0.3
      interpretation_weight: 0.2
      correctness_weight: 0.5
      
      # 动态奖励配置
      dynamic_reward: true
      early_phase: 30
      middle_phase: 100
      initial_scale: 0.5
      mid_scale: 0.8
      full_scale: 1.0
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "A tool for calculating the reward of gsm8k. (1.0 if your answer is correct, 0.0 if your answer is incorrect)"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The model's answer to the GSM8K math problem, must be a digits"
            reasoning:
              type: "string"
              description: "The reasoning process used to arrive at the answer"
          required: ["answer"]
```

### 步骤 4：修改 AsyncSGLangRollout 类

修改 `AsyncSGLangRollout` 类中的工具调用处理逻辑，支持增强的奖励机制。

```python
# 文件路径：verl/workers/rollout/sglang_rollout/async_sglang_rollout.py

# 在 _async_rollout_a_request 方法中，修改工具调用部分

# 原代码
if _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
    if _req.messages[-1].tool_calls is not None:
        parsed_tool_calls = _req.messages[-1].tool_calls
        tool_call_results = await asyncio.gather(
            *[
                self._tool_map[tool_call.function.name].execute(
                    _req.request_id,
                    tool_call.function.arguments,
                    **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
                )
                for tool_call in parsed_tool_calls
            ]
        )
        for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results):
            _req.add_tool_response_message(self.tokenizer, resp, format=self.config.multi_turn.format)
            if len(_req.input_ids) >= self.config.max_model_len:
                break
        if len(_req.input_ids) >= self.config.max_model_len:
            finish_reason_type = FinishReasonTypeEnum.STOP
            break
        _req.state = AsyncRolloutRequestStateEnum.RUNNING
    else:
        raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")

# 修改为
if _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
    if _req.messages[-1].tool_calls is not None:
        parsed_tool_calls = _req.messages[-1].tool_calls
        
        # 获取上下文信息
        context = _req.get_context()
        
        # 获取训练步骤信息
        training_step = _req.get_training_step()
        
        # 准备工具调用
        tool_call_tasks = []
        for tool_call in parsed_tool_calls:
            tool_name = tool_call.function.name
            tool = self._tool_map[tool_name]
            
            # 准备执行参数
            execute_kwargs = _req.tools_kwargs[tool_name].get("execute_kwargs", {})
            
            # 添加上下文和训练步骤信息
            execute_kwargs.update({
                "context": context,
                "training_step": training_step,
                "difficulty": _req.get_difficulty(),
            })
            
            # 创建工具调用任务
            tool_call_tasks.append(
                tool.execute(
                    _req.request_id,
                    tool_call.function.arguments,
                    **execute_kwargs,
                )
            )
        
        # 执行所有工具调用
        tool_call_results = await asyncio.gather(*tool_call_tasks)
        
        # 处理工具调用结果
        for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results):
            # 添加工具响应消息
            _req.add_tool_response_message(self.tokenizer, resp, format=self.config.multi_turn.format)
            
            # 记录详细指标
            _req.add_metrics(tool_call.function.name, metrics)
            
            if len(_req.input_ids) >= self.config.max_model_len:
                break
                
        if len(_req.input_ids) >= self.config.max_model_len:
            finish_reason_type = FinishReasonTypeEnum.STOP
            break
            
        _req.state = AsyncRolloutRequestStateEnum.RUNNING
    else:
        raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
```

### 步骤 5：扩展 AsyncRolloutRequest 类

扩展 `AsyncRolloutRequest` 类，添加对上下文、训练步骤和指标的支持。

```python
# 文件路径：verl/workers/rollout/schemas.py

# 在 AsyncRolloutRequest 类中添加新方法

def get_context(self) -> str:
    """获取请求的上下文信息"""
    # 从消息历史中提取上下文
    if self.messages:
        # 获取用户消息
        user_messages = [msg.content for msg in self.messages if msg.role == "user"]
        return "\n".join(user_messages)
    return ""

def get_training_step(self) -> int:
    """获取当前训练步骤"""
    # 从请求元数据中获取训练步骤信息
    return getattr(self, "training_step", 0)

def get_difficulty(self) -> float:
    """获取任务难度"""
    # 从请求元数据中获取难度信息，默认为中等难度
    return getattr(self, "difficulty", 1.0)

def add_metrics(self, tool_name: str, metrics: dict) -> None:
    """添加工具调用指标"""
    if not hasattr(self, "metrics"):
        self.metrics = {}
    
    if tool_name not in self.metrics:
        self.metrics[tool_name] = []
        
    self.metrics[tool_name].append(metrics)
```

### 步骤 6：修改训练脚本

修改训练脚本，支持增强的工具配置。

```bash
# 文件路径：examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_4xgpu_enhanced.sh

# 在原有脚本基础上修改，使用增强的工具配置
actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/enhanced_tool_config.yaml"
```

## 三、实施计划

1. **准备阶段**（1-2天）：
   - 创建 `enhanced_base_tool.py` 文件
   - 创建 `enhanced_tool_config.yaml` 配置文件
   - 准备单元测试

2. **开发阶段**（3-5天）：
   - 实现 `EnhancedBaseTool` 类
   - 修改 `Gsm8kTool` 类
   - 扩展 `AsyncRolloutRequest` 类
   - 修改 `AsyncSGLangRollout` 类

3. **测试阶段**（2-3天）：
   - 单元测试各组件功能
   - 集成测试工具调用流程
   - 验证奖励计算逻辑

4. **训练阶段**（5-7天）：
   - 使用增强的奖励机制进行小规模训练
   - 分析训练效果
   - 调整奖励参数

5. **评估阶段**（2-3天）：
   - 与基线模型对比性能
   - 分析模型行为变化
   - 总结经验和改进方向

## 四、预期效果

1. **更精细的学习信号**：
   - 模型能够获得关于工具选择、参数构造、结果解释等方面的精细反馈
   - 减少无效工具调用，提高工具使用效率

2. **更稳定的训练过程**：
   - 动态奖励调整机制使模型在训练初期获得更多正向反馈
   - 随着训练进行，逐渐提高标准，引导模型学习更复杂的行为

3. **更强的泛化能力**：
   - 通过评估工具调用的多个维度，提高模型在未见场景中的表现
   - 鼓励模型学习通用的工具使用策略，而不仅仅是特定任务的解决方案

4. **更好的推理能力**：
   - 通过评估思考过程和结果解释，引导模型学习更好的推理能力
   - 鼓励模型生成可解释的工具调用过程

## 五、潜在挑战与解决方案

1. **计算开销增加**：
   - **挑战**：更精细的奖励计算可能增加训练时间
   - **解决方案**：优化奖励计算逻辑，考虑使用缓存机制减少重复计算

2. **奖励设计复杂性**：
   - **挑战**：多维度奖励可能引入更多超参数，增加调优难度
   - **解决方案**：提供合理的默认值，使用自动化调参技术寻找最优配置

3. **不同工具类型适配**：
   - **挑战**：不同类型的工具可能需要不同的奖励计算逻辑
   - **解决方案**：设计灵活的接口，允许不同工具类型定制自己的奖励计算方法

4. **与现有代码集成**：
   - **挑战**：修改现有代码可能引入兼容性问题
   - **解决方案**：采用渐进式修改策略，确保向后兼容性，提供充分的测试覆盖

## 六、结论

通过实施上述优化方案，sglang_multiturn 工具奖励机制将获得显著提升，提供更精细、更有效的学习信号，引导模型学习更好的工具使用策略。这些改进将使模型能够更好地理解何时以及如何使用工具，以及如何解释和利用工具输出来解决问题。
