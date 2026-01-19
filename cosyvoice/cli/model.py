# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from cosyvoice.utils.common import TrtContextWrapper

# 流式输出优化：使用条件变量替代轮询，减少CPU占用和延迟
class StreamingTokenBuffer:
    """
    高效的流式Token缓冲区
    
    优化点：
    1. 使用条件变量通知而非轮询
    2. 减少锁持有时间
    3. 提供零拷贝的token访问方式
    4. 支持增量获取tokens（避免重复复制历史数据）
    """
    
    def __init__(self):
        self.tokens = []
        self.is_finished = False
        self.condition = threading.Condition()
        # 优化：记录上次返回的位置，支持增量获取
        self._last_read_pos = 0
    
    def append(self, token):
        """添加token并通知等待的消费者"""
        with self.condition:
            self.tokens.append(token)
            self.condition.notify()
    
    def extend(self, tokens):
        """批量添加tokens"""
        with self.condition:
            self.tokens.extend(tokens)
            self.condition.notify()
    
    def finish(self):
        """标记生产完成"""
        with self.condition:
            self.is_finished = True
            self.condition.notify_all()
    
    def wait_for_tokens(self, required_count: int, timeout: float = 0.05) -> bool:
        """
        等待直到有足够的tokens可用或超时
        Args:
            required_count: 需要的token数量
            timeout: 最大等待时间（秒），默认50ms以减少延迟
        Returns:
            是否有足够的tokens
        """
        with self.condition:
            # 使用条件变量等待，比sleep更高效
            end_time = time.time() + timeout
            while len(self.tokens) < required_count and not self.is_finished:
                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)
            return len(self.tokens) >= required_count
    
    def get_tokens(self):
        """获取所有tokens的副本（兼容旧接口）"""
        with self.condition:
            return list(self.tokens)
    
    def get_tokens_slice(self, start: int, end: int):
        """
        获取指定范围的tokens（零拷贝访问底层列表的切片）
        
        优化：直接返回切片，避免复制整个列表
        注意：返回的是切片视图，调用者不应修改
        """
        with self.condition:
            return self.tokens[start:end]
    
    def get_token_count(self) -> int:
        """获取当前token数量（无需复制）"""
        with self.condition:
            return len(self.tokens)
    
    def get_new_tokens(self):
        """
        获取自上次调用以来新增的tokens
        
        优化：用于增量处理场景，避免重复处理历史tokens
        """
        with self.condition:
            new_tokens = self.tokens[self._last_read_pos:]
            self._last_read_pos = len(self.tokens)
            return new_tokens
    
    def pop_tokens(self, count: int):
        """移除并返回前count个tokens"""
        with self.condition:
            popped = self.tokens[:count]
            self.tokens = self.tokens[count:]
            # 更新读取位置
            self._last_read_pos = max(0, self._last_read_pos - count)
            return popped
    
    def __len__(self):
        with self.condition:
            return len(self.tokens)


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = []
        
        # 修复流式输出开头语气词问题：跳过每段文本第一个流式块的开头部分
        # 设置为0.35秒，可以通过set_stream_first_chunk_skip方法调整
        # 设置为0则不跳过
        self.stream_first_chunk_skip_seconds = 0.35
        
        # 修复音频结尾语气词问题：跳过音频末尾部分
        # 设置为0.1秒，可以通过set_end_chunk_skip方法调整
        self.end_chunk_skip_seconds = 0.1
    
    def set_stream_first_chunk_skip(self, skip_seconds: float):
        """
        设置流式输出时第一个音频块开头要跳过的时长（秒）
        
        用于消除每段文本开始时的语气词（如"啊"、"嗯"等）
        
        Args:
            skip_seconds: 要跳过的秒数
                - 0: 不跳过（可能有语气词）
                - 0.2: 跳过0.2秒（轻微修正）
                - 0.35: 默认值，跳过0.35秒（推荐）
                - 0.5: 跳过0.5秒（强修正，可能切掉部分内容）
        """
        assert 0 <= skip_seconds <= 1.0, "skip_seconds应该在0-1.0之间"
        self.stream_first_chunk_skip_seconds = skip_seconds
    
    def set_end_chunk_skip(self, skip_seconds: float):
        """
        设置音频结尾要跳过的时长（秒）
        
        用于消除音频结束时的语气词（如"嗯"、"哼"等）
        
        Args:
            skip_seconds: 要跳过的秒数
                - 0: 不跳过
                - 0.1: 默认值，跳过0.1秒（推荐）
                - 0.2: 跳过0.2秒（强修正）
        """
        assert 0 <= skip_seconds <= 0.5, "skip_seconds应该在0-0.5之间"
        self.end_chunk_skip_seconds = skip_seconds

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device, weights_only=True), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device, weights_only=True), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        cur_silent_token_num, max_silent_token_num = 0, 5
        with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            if isinstance(text, Generator):
                assert (self.__class__.__name__ != 'CosyVoiceModel') and not hasattr(self.llm, 'vllm'), 'streaming input text is only implemented for CosyVoice2/3 and do not support vllm!'
                token_generator = self.llm.inference_bistream(text=text,
                                                              prompt_text=prompt_text.to(self.device),
                                                              prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                              prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                              prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                              embedding=llm_embedding.to(self.device))
            else:
                token_generator = self.llm.inference(text=text.to(self.device),
                                                     text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_text=prompt_text.to(self.device),
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                     embedding=llm_embedding.to(self.device),
                                                     uuid=uuid)
            
            # 优化：使用StreamingTokenBuffer的条件变量通知机制
            token_buffer = self.tts_speech_token_dict.get(uuid)
            use_buffer = isinstance(token_buffer, StreamingTokenBuffer)
            
            for i in token_generator:
                if i in self.silent_tokens:
                    cur_silent_token_num += 1
                    if cur_silent_token_num > max_silent_token_num:
                        continue
                else:
                    cur_silent_token_num = 0
                
                if use_buffer:
                    token_buffer.append(i)
                else:
                    self.tts_speech_token_dict[uuid].append(i)
        
        # 通知生产完成
        if use_buffer:
            token_buffer.finish()
        self.llm_end_dict[uuid] = True

    def vc_job(self, source_speech_token, uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist()
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_token=prompt_token.to(self.device),
                                                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_feat=prompt_feat.to(self.device),
                                                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                                      embedding=embedding.to(self.device),
                                                                      flow_cache=self.flow_cache_dict[uuid])

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        
        # 刷新 HiFiGAN 的随机状态，确保每次新的合成请求都有不同的音频微妙特征
        # 这对于 Causal 模式尤其重要，可以避免不同合成请求之间的音频特征相似性
        if hasattr(self.hift, 'refresh_random_state'):
            self.hift.refresh_random_state()
        
        # 优化：使用StreamingTokenBuffer替代普通列表，支持条件变量通知
        with self.lock:
            if stream:
                self.tts_speech_token_dict[this_uuid] = StreamingTokenBuffer()
            else:
                self.tts_speech_token_dict[this_uuid] = []
            self.llm_end_dict[this_uuid] = False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        
        if stream is True:
            token_buffer = self.tts_speech_token_dict[this_uuid]
            token_hop_len = self.token_min_hop_len
            
            # 优化：使用条件变量等待，减少CPU占用
            wait_timeout = 0.05
            
            # 修复流式输出开头语气词问题：跟踪是否为第一个流式块
            is_first_chunk = True
            # 计算要跳过的采样数（CosyVoice1的采样率为22050Hz）
            sample_rate = 22050
            first_chunk_skip_samples = int(self.stream_first_chunk_skip_seconds * sample_rate)
            
            while True:
                required_tokens = token_hop_len + self.token_overlap_len
                
                # 优化：使用条件变量等待，而非time.sleep轮询
                token_buffer.wait_for_tokens(required_tokens, timeout=wait_timeout)
                
                current_tokens = token_buffer.get_tokens()
                current_token_len = len(current_tokens)
                
                if current_token_len >= required_tokens:
                    this_tts_speech_token = torch.tensor(current_tokens[:required_tokens]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    
                    # 修复流式输出开头语气词：跳过第一个流式块的开头部分
                    if is_first_chunk and first_chunk_skip_samples > 0:
                        if this_tts_speech.shape[1] > first_chunk_skip_samples:
                            this_tts_speech = this_tts_speech[:, first_chunk_skip_samples:]
                        else:
                            # 如果第一个块太短，跳过整个块，不输出
                            token_buffer.pop_tokens(token_hop_len)
                            token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                            is_first_chunk = False
                            continue
                        is_first_chunk = False
                    
                    yield {'tts_speech': this_tts_speech.cpu()}
                    
                    # 移除已处理的tokens
                    token_buffer.pop_tokens(token_hop_len)
                    
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                
                # 检查是否应该退出循环
                if token_buffer.is_finished and len(token_buffer) < required_tokens:
                    break
            
            p.join()
            
            # 处理剩余的tokens
            remaining_tokens = token_buffer.get_tokens()
            if len(remaining_tokens) > 0:
                this_tts_speech_token = torch.tensor(remaining_tokens).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                 prompt_token=flow_prompt_speech_token,
                                                 prompt_feat=prompt_speech_feat,
                                                 embedding=flow_embedding,
                                                 uuid=this_uuid,
                                                 finalize=True)
                
                # 如果所有之前的块都被跳过了，第一个有效块仍然需要处理
                if is_first_chunk and first_chunk_skip_samples > 0:
                    if this_tts_speech.shape[1] > first_chunk_skip_samples:
                        this_tts_speech = this_tts_speech[:, first_chunk_skip_samples:]
                
                yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # 非流式模式：等待所有tokens生成完成
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            
            # 非流式模式：裁剪开头和结尾的语气词
            # CosyVoice1的采样率为22050Hz
            sample_rate = 22050
            start_skip_samples = int(self.stream_first_chunk_skip_seconds * sample_rate)
            end_skip_samples = int(self.end_chunk_skip_seconds * sample_rate)
            
            # 确保裁剪后还有足够的音频数据
            total_samples = this_tts_speech.shape[1]
            if start_skip_samples + end_skip_samples < total_samples:
                if end_skip_samples > 0:
                    this_tts_speech = this_tts_speech[:, start_skip_samples:-end_skip_samples]
                else:
                    this_tts_speech = this_tts_speech[:, start_skip_samples:]
            
            yield {'tts_speech': this_tts_speech.cpu()}
        
        # 清理资源
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        
        # 优化：减少不必要的显存清理调用


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        # NOTE must matching training static_chunk_size
        # 优化：增大token_hop_len可以减少flow推理次数，从而降低RTF
        # 默认35（从25提升），可以根据需要调整到40-50以进一步降低RTF（但会增加首包延迟）
        # 25: 低延迟（首包约0.5s），35: 平衡（首包约0.7s），50: 低RTF（首包约1s）
        self.token_hop_len = 35  # 优化：从25增大到35，减少约30%的flow推理次数
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        # 优化：为Flow创建独立的CUDA stream，允许与LLM并行计算
        self.flow_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = []
        
        # 优化参数：flow的diffusion步数，减少可以降低RTF但可能略微影响音质
        # 10: 高质量, 8: 平衡(默认), 6: 低延迟
        self.flow_n_timesteps = 8  # 优化：从10减少到8，降低约20%的diffusion计算量
        
        # 修复流式输出开头语气词问题：跳过每段文本第一个流式块的开头部分
        # 设置为0.35秒，可以通过set_stream_first_chunk_skip方法调整
        # 设置为0则不跳过
        self.stream_first_chunk_skip_seconds = 0.1
        
        # 修复音频结尾语气词问题：跳过音频末尾部分
        # 设置为0.1秒，可以通过set_end_chunk_skip方法调整
        self.end_chunk_skip_seconds = 0.1
        
        # 优化：自适应RTF机制 - 根据当前RTF动态调整参数
        self._rtf_history = []  # 记录最近的RTF值
        self._adaptive_rtf_enabled = False  # 默认关闭自适应RTF（滑动窗口更有效）
        self._min_token_hop_len = 25  # 最小token_hop_len
        self._max_token_hop_len = 60  # 最大token_hop_len
        
        # ========== 核心优化：滑动窗口机制 ==========
        # 限制每次flow推理处理的token数量，避免RTF随文本长度线性增长
        # 这是解决长文本RTF升高问题的关键！
        # 
        # 原理：Flow模型在流式模式下，每次只需要：
        #   1. prompt tokens（用于speaker embedding）
        #   2. 最近生成的tokens（用于上下文连贯性）
        # 不需要处理全部历史tokens
        #
        # 参数说明：
        # - max_context_tokens: 滑动窗口大小，每次最多处理这么多tokens
        #   值越大：上下文越完整，音质越好，但RTF越高
        #   值越小：RTF越低，但可能影响音频连贯性
        #   推荐范围：100-200
        self.max_context_tokens = 150  # 滑动窗口大小
        self.use_sliding_window = True  # 是否启用滑动窗口
        
        # 优化：尝试使用torch.compile加速模型（PyTorch 2.0+）
        self._try_compile_models()

    def _try_compile_models(self):
        """尝试使用torch.compile加速模型（仅PyTorch 2.0+支持）"""
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                # 使用reduce-overhead模式，专门优化推理场景
                # 注意：首次推理会较慢（编译），后续会显著加速
                # self.flow.decoder = torch.compile(self.flow.decoder, mode='reduce-overhead')
                # self.hift = torch.compile(self.hift, mode='reduce-overhead')
                # 暂时禁用torch.compile，因为可能与流式推理有兼容性问题
                pass
        except Exception as e:
            # torch.compile可能不可用或失败，静默忽略
            pass
    
    def set_flow_timesteps(self, n_timesteps: int):
        """
        设置Flow的diffusion步数
        
        Args:
            n_timesteps: diffusion步数
                - 10: 高质量(默认)
                - 8: 平衡模式，RTF降低约20%，音质几乎无损
                - 6: 低延迟模式，RTF降低约40%，音质略有下降
        """
        assert 4 <= n_timesteps <= 10, "n_timesteps应该在4-10之间"
        self.flow_n_timesteps = n_timesteps
    
    def set_token_hop_len(self, token_hop_len: int):
        """
        设置流式处理的token跳跃长度
        
        增大此值可以减少flow推理次数，从而降低RTF，但会增加首包延迟。
        
        Args:
            token_hop_len: token跳跃长度
                - 25: 默认值，首包延迟约0.5s
                - 35: 平衡模式，首包延迟约0.7s，RTF降低约15%
                - 50: 低RTF模式，首包延迟约1s，RTF降低约25%
        """
        assert 20 <= token_hop_len <= 100, "token_hop_len应该在20-100之间"
        self.token_hop_len = token_hop_len
    
    def set_stream_first_chunk_skip(self, skip_seconds: float):
        """
        设置流式输出时第一个音频块开头要跳过的时长（秒）
        
        用于消除每段文本开始时的语气词（如"啊"、"嗯"等）
        
        Args:
            skip_seconds: 要跳过的秒数
                - 0: 不跳过（可能有语气词）
                - 0.2: 跳过0.2秒（轻微修正）
                - 0.35: 默认值，跳过0.35秒（推荐）
                - 0.5: 跳过0.5秒（强修正，可能切掉部分内容）
        """
        assert 0 <= skip_seconds <= 1.0, "skip_seconds应该在0-1.0之间"
        self.stream_first_chunk_skip_seconds = skip_seconds
    
    def set_end_chunk_skip(self, skip_seconds: float):
        """
        设置音频结尾要跳过的时长（秒）
        
        用于消除音频结束时的语气词（如"嗯"、"哼"等）
        
        Args:
            skip_seconds: 要跳过的秒数
                - 0: 不跳过
                - 0.1: 默认值，跳过0.1秒（推荐）
                - 0.2: 跳过0.2秒（强修正）
        """
        assert 0 <= skip_seconds <= 0.5, "skip_seconds应该在0-0.5之间"
        self.end_chunk_skip_seconds = skip_seconds
    
    def set_adaptive_rtf(self, enabled: bool = True, min_hop: int = 25, max_hop: int = 60):
        """
        设置自适应RTF调整
        
        当启用时，系统会根据当前RTF动态调整token_hop_len：
        - 如果RTF > 1（生成速度跟不上播放），增大token_hop_len减少推理次数
        - 如果RTF < 0.7（生成速度远超播放），减小token_hop_len提高实时性
        
        Args:
            enabled: 是否启用自适应调整
            min_hop: 最小token_hop_len（默认25）
            max_hop: 最大token_hop_len（默认60）
        """
        self._adaptive_rtf_enabled = enabled
        self._min_token_hop_len = min_hop
        self._max_token_hop_len = max_hop
    
    def set_sliding_window(self, enabled: bool = True, max_context: int = 150):
        """
        设置滑动窗口优化
        
        这是解决长文本RTF线性增长问题的核心优化！
        
        原理：每次flow推理只处理最近的max_context个tokens，而不是全部历史tokens。
        这将flow推理的计算复杂度从O(n)降低到O(1)，使RTF保持稳定。
        
        Args:
            enabled: 是否启用滑动窗口（默认True，强烈推荐开启）
            max_context: 滑动窗口大小，即每次处理的最大token数量
                - 100: 低RTF模式，适合长文本
                - 150: 平衡模式（默认推荐）
                - 200: 高质量模式，上下文更完整
        """
        self.use_sliding_window = enabled
        self.max_context_tokens = max_context
        from cosyvoice.utils.file_utils import logging
        if enabled:
            logging.info(f'滑动窗口优化已启用: max_context_tokens={max_context}')
    
    def _update_adaptive_token_hop_len(self, rtf: float):
        """
        根据RTF更新token_hop_len
        
        自适应策略：
        - RTF > 1.2: 增大token_hop_len 20%
        - RTF > 1.0: 增大token_hop_len 10%
        - RTF < 0.5: 减小token_hop_len 10%
        - RTF < 0.7: 保持或略微减小
        """
        if not self._adaptive_rtf_enabled:
            return
        
        # 记录RTF历史（保留最近10个值）
        self._rtf_history.append(rtf)
        if len(self._rtf_history) > 10:
            self._rtf_history.pop(0)
        
        # 使用平均RTF来避免抖动
        if len(self._rtf_history) < 3:
            return
        
        avg_rtf = sum(self._rtf_history[-5:]) / min(5, len(self._rtf_history))
        
        current_hop = self.token_hop_len
        
        if avg_rtf > 1.2:
            # RTF严重超标，大幅增大token_hop_len
            new_hop = min(self._max_token_hop_len, int(current_hop * 1.2))
        elif avg_rtf > 1.0:
            # RTF超标，适度增大
            new_hop = min(self._max_token_hop_len, int(current_hop * 1.1))
        elif avg_rtf < 0.5:
            # RTF很低，可以减小token_hop_len提高实时性
            new_hop = max(self._min_token_hop_len, int(current_hop * 0.9))
        else:
            # RTF正常，保持不变
            new_hop = current_hop
        
        if new_hop != current_hop:
            self.token_hop_len = new_hop
            from cosyvoice.utils.file_utils import logging
            logging.info(f'自适应RTF调整: token_hop_len {current_hop} -> {new_hop} (avg_rtf={avg_rtf:.2f})')
    
    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_vllm(self, model_dir):
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        from vllm import EngineArgs, LLMEngine
        engine_args = EngineArgs(model=model_dir,
                                 skip_tokenizer_init=True,
                                 enable_prompt_embeds=True,
                                 gpu_memory_utilization=0.3)
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        # 使用可配置的n_timesteps参数
        n_timesteps = getattr(self, 'flow_n_timesteps', 10)
        
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize,
                                             n_timesteps=n_timesteps)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        
        # 刷新 HiFiGAN 的随机状态，确保每次新的合成请求都有不同的音频微妙特征
        # 这对于 Causal 模式（CosyVoice3）尤其重要，可以避免不同合成请求之间的音频特征相似性
        if hasattr(self.hift, 'refresh_random_state'):
            self.hift.refresh_random_state()
        
        # 优化：使用StreamingTokenBuffer替代普通列表，支持条件变量通知
        with self.lock:
            if stream:
                self.tts_speech_token_dict[this_uuid] = StreamingTokenBuffer()
            else:
                self.tts_speech_token_dict[this_uuid] = []
            self.llm_end_dict[this_uuid] = False
            self.hift_cache_dict[this_uuid] = None
        
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        
        if stream is True:
            token_buffer = self.tts_speech_token_dict[this_uuid]
            token_offset = 0
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
            
            # 优化：使用条件变量等待，超时时间设为50ms，在资源紧张时更加高效
            wait_timeout = 0.05
            
            # 修复流式输出开头语气词问题：跟踪是否为第一个流式块
            is_first_chunk = True
            # 计算要跳过的采样数（假设采样率为24000Hz）
            sample_rate = 24000  # CosyVoice2/3的默认采样率
            first_chunk_skip_samples = int(self.stream_first_chunk_skip_seconds * sample_rate)
            
            # 优化：预分配tensor缓冲区，减少内存分配次数
            # 估计最大token数量（基于文本长度，每个字符约20个tokens）
            max_estimated_tokens = 4000  # 足够处理大多数请求
            token_tensor_buffer = None
            
            # 优化：RTF监测变量
            chunk_start_time = time.time()
            total_audio_duration = 0.0
            
            while True:
                this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
                required_tokens = token_offset + this_token_hop_len + self.flow.pre_lookahead_len
                
                # 优化：使用条件变量等待，而非time.sleep轮询
                # 当有足够的tokens时立即被唤醒，减少延迟
                token_buffer.wait_for_tokens(required_tokens, timeout=wait_timeout)
                
                # 优化：使用get_token_count()避免复制整个列表来获取长度
                current_token_len = token_buffer.get_token_count()
                
                if current_token_len >= required_tokens:
                    # 优化：使用get_tokens_slice()只获取需要的部分，减少内存复制
                    end_idx = token_offset + this_token_hop_len + self.flow.pre_lookahead_len
                    
                    # ========== 核心优化：滑动窗口 ==========
                    # 只传入最近的max_context_tokens个tokens，而不是全部历史
                    # 这将RTF从O(n)降低到O(1)，是解决长文本RTF升高的关键！
                    if self.use_sliding_window and end_idx > self.max_context_tokens:
                        # 计算窗口起始位置：保留最近的max_context_tokens个tokens
                        window_start = end_idx - self.max_context_tokens
                        token_slice = token_buffer.get_tokens_slice(window_start, end_idx)
                        # 调整token_offset：相对于窗口起始位置
                        effective_token_offset = token_offset - window_start
                    else:
                        # tokens数量未超过窗口大小，使用全部
                        token_slice = token_buffer.get_tokens_slice(0, end_idx)
                        effective_token_offset = token_offset
                    
                    this_tts_speech_token = torch.tensor(token_slice, dtype=torch.int32).unsqueeze(dim=0)
                    
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     token_offset=effective_token_offset,
                                                     uuid=this_uuid,
                                                     stream=stream,
                                                     finalize=False)
                    token_offset += this_token_hop_len
                    
                    # 修复流式输出开头语气词：跳过第一个流式块的开头部分
                    if is_first_chunk and first_chunk_skip_samples > 0:
                        if this_tts_speech.shape[1] > first_chunk_skip_samples:
                            this_tts_speech = this_tts_speech[:, first_chunk_skip_samples:]
                        else:
                            # 如果第一个块太短，跳过整个块，不输出
                            is_first_chunk = False
                            continue
                        is_first_chunk = False
                    
                    # 优化：计算当前chunk的RTF并进行自适应调整
                    chunk_end_time = time.time()
                    chunk_duration = chunk_end_time - chunk_start_time
                    audio_duration = this_tts_speech.shape[1] / sample_rate
                    total_audio_duration += audio_duration
                    
                    if chunk_duration > 0 and audio_duration > 0:
                        chunk_rtf = chunk_duration / audio_duration
                        # 调用自适应RTF调整
                        if hasattr(self, '_update_adaptive_token_hop_len'):
                            self._update_adaptive_token_hop_len(chunk_rtf)
                    
                    chunk_start_time = time.time()  # 重置计时器
                    
                    yield {'tts_speech': this_tts_speech.cpu()}
                
                # 检查是否应该退出循环（优化：使用is_finished属性而非方法调用）
                if token_buffer.is_finished and current_token_len < required_tokens:
                    break
            
            p.join()
            
            # 处理剩余的tokens - 优化：使用切片而非复制整个列表
            total_tokens = token_buffer.get_token_count()
            if total_tokens > token_offset:
                remaining_token_slice = token_buffer.get_tokens_slice(token_offset, total_tokens)
                this_tts_speech_token = torch.tensor(remaining_token_slice, dtype=torch.int32).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                 prompt_token=flow_prompt_speech_token,
                                                 prompt_feat=prompt_speech_feat,
                                                 embedding=flow_embedding,
                                                 token_offset=0,
                                                 uuid=this_uuid,
                                                 finalize=True)
                
                # 如果所有之前的块都被跳过了，第一个有效块仍然需要处理
                if is_first_chunk and first_chunk_skip_samples > 0:
                    if this_tts_speech.shape[1] > first_chunk_skip_samples:
                        this_tts_speech = this_tts_speech[:, first_chunk_skip_samples:]
                
                yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # 非流式模式：等待所有tokens生成完成
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=0,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            
            # 非流式模式：裁剪开头和结尾的语气词
            # CosyVoice2的采样率为24000Hz
            sample_rate = 24000
            start_skip_samples = int(self.stream_first_chunk_skip_seconds * sample_rate)
            end_skip_samples = int(self.end_chunk_skip_seconds * sample_rate)
            
            # 确保裁剪后还有足够的音频数据
            total_samples = this_tts_speech.shape[1]
            if start_skip_samples + end_skip_samples < total_samples:
                if end_skip_samples > 0:
                    this_tts_speech = this_tts_speech[:, start_skip_samples:-end_skip_samples]
                else:
                    this_tts_speech = this_tts_speech[:, start_skip_samples:]
            
            yield {'tts_speech': this_tts_speech.cpu()}
        
        # 清理资源
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        
        # 优化：减少不必要的显存清理调用，只在确实需要时执行
        # torch.cuda.empty_cache() 在高负载下会造成额外开销


class CosyVoice3Model(CosyVoice2Model):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        # NOTE must matching training static_chunk_size
        # 优化：增大token_hop_len可以减少flow推理次数
        # 从25增大到35，减少约30%的推理次数
        self.token_hop_len = 35
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.flow_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        # FSQ silent and breath token
        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
        
        # 优化参数：flow的diffusion步数
        # 从10减少到8，降低约20%的计算量
        self.flow_n_timesteps = 8
        
        # 修复流式输出开头语气词问题：跳过每段文本第一个流式块的开头部分
        # 设置为0.35秒，可以通过set_stream_first_chunk_skip方法调整
        self.stream_first_chunk_skip_seconds = 0.35
        
        # 修复音频结尾语气词问题：跳过音频末尾部分
        # 设置为0.1秒，可以通过set_end_chunk_skip方法调整
        self.end_chunk_skip_seconds = 0.1
        
        # 优化：自适应RTF机制（默认关闭，滑动窗口更有效）
        self._rtf_history = []
        self._adaptive_rtf_enabled = False
        self._min_token_hop_len = 25
        self._max_token_hop_len = 60
        
        # ========== 核心优化：滑动窗口机制 ==========
        # 与CosyVoice2Model相同的滑动窗口参数
        self.max_context_tokens = 150
        self.use_sliding_window = True
        
        # 尝试编译模型
        self._try_compile_models()

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        # 使用可配置的n_timesteps参数
        n_timesteps = getattr(self, 'flow_n_timesteps', 10)
        
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize,
                                             n_timesteps=n_timesteps)
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
            # append mel cache
            if self.hift_cache_dict[uuid] is not None:
                hift_cache_mel = self.hift_cache_dict[uuid]['mel']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                self.hift_cache_dict[uuid]['mel'] = tts_mel
            else:
                self.hift_cache_dict[uuid] = {'mel': tts_mel, 'speech_offset': 0}
            if speed != 1.0:
                assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            tts_speech = tts_speech[:, self.hift_cache_dict[uuid]['speech_offset']:]
            self.hift_cache_dict[uuid]['speech_offset'] += tts_speech.shape[1]
        return tts_speech
