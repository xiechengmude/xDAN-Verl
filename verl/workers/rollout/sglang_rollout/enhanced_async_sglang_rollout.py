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

import asyncio
import json
import logging
import os
from copy import deepcopy
from json.decoder import JSONDecodeError

import torch

from verl.tools.base_tool import BaseTool
from verl.tools.enhanced_base_tool import EnhancedBaseTool
from verl.workers.rollout.sglang_rollout.async_sglang_rollout import AsyncSGLangRollout
from verl.workers.rollout.sglang_rollout.async_rollout_request import AsyncRolloutRequest, FinishReasonTypeEnum

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnhancedAsyncSGLangRollout(AsyncSGLangRollout):
    """增强版SGLang异步回滚，支持精细化奖励计算"""

    async def _async_rollout_a_request(self, req: AsyncRolloutRequest, do_sample: bool = True, is_validate: bool = False, **kwargs) -> AsyncRolloutRequest:
        """
        处理异步回滚请求，支持精细化奖励计算
        
        Args:
            req: 异步回滚请求
            do_sample: 是否进行采样
            is_validate: 是否为验证模式
            
        Returns:
            处理后的异步回滚请求
        """
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None

        # 记录训练步骤信息（如果有）
        training_step = kwargs.get("training_step", 0)
        
        current_turns = 0
        while current_turns < self.config.multi_turn.max_turns:
            if _req.state == AsyncRolloutRequest.StateEnum.PENDING:
                if _req.tools is not None:
                    tool_creation_coroutines = []
                    for tool_schema in _req.tools:
                        tool = self._tool_map[tool_schema.function.name]
                        create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
                        
                        # 对于增强型工具，添加训练步骤信息
                        if isinstance(tool, EnhancedBaseTool):
                            create_kwargs["training_step"] = training_step
                            
                        tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))
                    await asyncio.gather(*tool_creation_coroutines)
                _req.state = AsyncRolloutRequest.StateEnum.RUNNING
            elif _req.state == AsyncRolloutRequest.StateEnum.TOOL_CALLING:
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
                        
                        # 保存工具调用的详细反馈信息
                        if hasattr(_req, "tool_feedback") and metrics:
                            if _req.tool_feedback is None:
                                _req.tool_feedback = {}
                            _req.tool_feedback[tool_call.function.name] = metrics
                            
                        if len(_req.input_ids) >= self.config.max_model_len:
                            break
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequest.StateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequest.StateEnum.RUNNING:
                generation_prompt = _req.get_generation_prompt(self.tokenizer)
                if not do_sample:
                    kwargs = dict(
                        n=1,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                        repetition_penalty=1.0,
                        temperature=0,
                        top_p=1,
                        top_k=-1,
                        ignore_eos=False,
                        min_new_tokens=0,
                        max_new_tokens=self.config.response_length,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=True,
                    )
                elif is_validate:
                    # TODO: try **
                    kwargs = {
                        "top_k": self.config.val_kwargs.top_k,
                        "top_p": self.config.val_kwargs.top_p,
                        "temperature": self.config.val_kwargs.temperature,
                        "n": 1,  # if validate, already repeat in ray_trainer
                    }
                if "n" not in kwargs or kwargs["n"] > 1:  # group size is supported in preprocess
                    kwargs["n"] = 1
                # users can customize different sampling_params at different run
                with self.update_sampling_params(**kwargs):
                    output = await self._engine.async_generate(
                        prompt=generation_prompt,
                        sampling_params=self.sampling_params,
                        return_logprob=False,
                    )

                content = output["text"]
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.tokenizer, content, already_over_long=True, format=self.config.multi_turn.format)
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequest.StateEnum.TOOL_CALLING
                        try:
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except JSONDecodeError:
                            normed_content = content
                            tool_calls = []
                        except AttributeError:
                            normed_content = content
                            tool_calls = []
                        parsed_tool_calls = [
                            AsyncRolloutRequest.OpenAIFunctionToolCall(
                                id=str(tool_call.tool_index),
                                function=AsyncRolloutRequest.OpenAIFunctionParsedSchema(name=tool_call.name, arguments=tool_call.parameters),
                            )
                            for tool_call in tool_calls
                        ]
                        if len(parsed_tool_calls) > 0:
                            _req.add_assistant_message(
                                self.tokenizer,
                                normed_content,
                                tool_calls=parsed_tool_calls,
                                format=self.config.multi_turn.format,
                            )
                        else:
                            _req.add_assistant_message(self.tokenizer, content, format=self.config.multi_turn.format)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequest.StateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(self.tokenizer, content, format=self.config.multi_turn.format)
                        break

        if current_turns >= self.config.multi_turn.max_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # 计算每个工具的奖励
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            # 计算奖励
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            
            # 获取详细反馈（如果有）
            feedback = {}
            if hasattr(_req, "tool_feedback") and _req.tool_feedback and name in _req.tool_feedback:
                feedback = _req.tool_feedback[name]
                
            # 释放工具实例
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            
            return name, reward, feedback

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
            
        tool_results = await asyncio.gather(*tool_reward_tasks)
        
        # 处理奖励和反馈
        tool_reward_scores = {}
        tool_feedbacks = {}
        
        for name, reward, feedback in tool_results:
            tool_reward_scores[name] = reward
            if feedback:
                tool_feedbacks[name] = feedback
                
        # 完成请求
        _req.finalize(self.tokenizer, tool_reward_scores, finish_reason_type)
        
        # 保存详细反馈（如果有）
        if tool_feedbacks:
            _req.tool_feedbacks = tool_feedbacks

        return _req
