# RTF优化指南 - 确保与其他服务一起运行时RTF保持在1以下

## 问题描述

当CosyVoice TTS服务单独运行时，RTF（实时因子）能稳定在1以下，但与其他服务一起运行时，RTF会上升到1-2之间。本文档提供了全面的优化方案，确保即使在资源竞争环境下也能保持RTF < 1。

## 优化措施总览

### 1. 服务器配置优化

#### 1.1 uvicorn配置
- **workers参数**: 默认设置为1（避免多进程导致的GPU内存问题）
- **事件循环**: 使用`uvloop`替代默认事件循环，提升性能
- **日志级别**: 设置为`info`，减少日志开销

#### 1.2 并发控制
- **最大并发请求数**: 默认4，可通过`--max_concurrent_requests`参数调整
- **请求限流中间件**: 自动拒绝超过并发限制的请求，返回503状态码
- **健康检查端点**: 不受并发限制影响

### 2. GPU内存管理优化

#### 2.1 CUDA设置优化
- **内存分配策略**: 设置`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`，优化内存碎片
- **cudnn自动调优**: 启用`torch.backends.cudnn.benchmark = True`
- **内存池**: 启动时清理CUDA缓存

#### 2.2 vLLM GPU内存利用率
- **默认值**: 从0.2提升到0.4（可根据实际情况调整）
- **可配置**: 通过`--gpu_memory_utilization`参数调整（范围0.1-0.9）
- **建议值**: 
  - 单独运行: 0.4-0.5
  - 与其他服务一起运行: 0.3-0.4

#### 2.3 TensorRT并发
- **默认值**: 2个并发上下文
- **可配置**: 通过`--trt_concurrent`参数调整
- **建议**: 根据GPU内存情况调整，通常2-4个

### 3. 流式推理优化

#### 3.1 Sleep间隔优化
- **优化前**: 0.1秒
- **优化后**: 0.005秒（减少20倍延迟）
- **动态调整**: 当连续多次检查没有新token时，适当增加sleep时间，避免CPU空转

#### 3.2 Token清理优化
- **清理阈值**: 从8个hop降低到6个hop，更频繁地清理已处理的token
- **内存管理**: 定期清理已处理的token，避免列表过长导致切片操作变慢

#### 3.3 空检查优化
- **最大空检查次数**: 20次
- **动态sleep**: 超过最大空检查次数后，增加sleep时间

### 4. 请求处理优化

#### 4.1 GZip压缩
- 启用GZip中间件，减少网络传输时间
- 最小压缩大小: 1000字节

#### 4.2 性能监控
- **首包延迟监控**: 记录并日志输出首包延迟
- **总耗时监控**: 记录每个请求的总处理时间
- **GPU内存监控**: 健康检查端点返回GPU内存使用情况

## 使用方法

### 基础启动（推荐配置）

```bash
python tts_api_server.py \
    --model_dir models/Fun-CosyVoice3-0.5B-2512 \
    --load_vllm \
    --fp16 \
    --gpu_memory_utilization 0.35 \
    --trt_concurrent 2 \
    --max_concurrent_requests 4
```

### 与其他服务一起运行时的推荐配置

```bash
# 降低GPU内存利用率，避免与其他服务竞争
python tts_api_server.py \
    --model_dir models/Fun-CosyVoice3-0.5B-2512 \
    --load_vllm \
    --fp16 \
    --gpu_memory_utilization 0.3 \
    --trt_concurrent 2 \
    --max_concurrent_requests 3 \
    --workers 1
```

### 环境变量配置

```bash
export MODEL_DIR=models/Fun-CosyVoice3-0.5B-2512
export LOAD_VLLM=true
export FP16=true
export GPU_MEMORY_UTILIZATION=0.35
export TRT_CONCURRENT=2
export MAX_CONCURRENT_REQUESTS=4

python tts_api_server.py
```

## 参数说明

| 参数 | 默认值 | 说明 | 推荐值（与其他服务一起运行） |
|------|--------|------|---------------------------|
| `--workers` | 1 | uvicorn worker进程数 | 1（避免GPU内存问题） |
| `--max_concurrent_requests` | 4 | 最大并发请求数 | 3-4 |
| `--gpu_memory_utilization` | 0.4 | vLLM GPU内存利用率 | 0.3-0.35 |
| `--trt_concurrent` | 2 | TensorRT并发上下文数 | 2 |
| `--fp16` | False | 启用FP16精度 | True（推荐） |
| `--load_vllm` | False | 启用vLLM加速 | True（推荐） |

## 性能调优建议

### 1. GPU内存充足时
```bash
--gpu_memory_utilization 0.4 \
--trt_concurrent 3 \
--max_concurrent_requests 5
```

### 2. GPU内存紧张时（与其他服务一起运行）
```bash
--gpu_memory_utilization 0.3 \
--trt_concurrent 2 \
--max_concurrent_requests 3
```

### 3. 高并发场景
```bash
--gpu_memory_utilization 0.35 \
--trt_concurrent 2 \
--max_concurrent_requests 4 \
--workers 1
```

## 监控和诊断

### 健康检查端点

```bash
curl http://localhost:8000/health
```

返回信息包括：
- 模型加载状态
- 当前并发请求数
- 最大并发请求数
- GPU内存使用情况（如果可用）

### 性能日志

服务器会记录以下性能指标：
- 首包延迟
- 总处理时间
- 数据块数量
- RTF（在模型内部日志中）

## 故障排查

### RTF仍然 > 1时的检查清单

1. **检查GPU内存使用**
   ```bash
   nvidia-smi
   ```
   确保GPU内存没有被其他服务占满

2. **检查并发请求数**
   ```bash
   curl http://localhost:8000/health
   ```
   如果`current_concurrent_requests`接近`max_concurrent_requests`，考虑增加限制或优化其他服务

3. **降低GPU内存利用率**
   ```bash
   --gpu_memory_utilization 0.25
   ```

4. **检查是否有其他GPU密集型服务**
   - 考虑使用`CUDA_VISIBLE_DEVICES`指定GPU
   - 或者降低其他服务的GPU使用率

5. **启用FP16精度**
   ```bash
   --fp16
   ```
   可以显著减少GPU内存使用和提升速度

## 最佳实践

1. **资源隔离**: 如果可能，为TTS服务分配专用的GPU
2. **监控RTF**: 定期检查RTF，确保保持在1以下
3. **渐进式调优**: 从保守配置开始，逐步增加并发和GPU利用率
4. **负载测试**: 在实际负载下测试，而不是仅测试单个请求
5. **日志分析**: 关注首包延迟和总处理时间的变化

## 预期效果

实施这些优化后，预期效果：
- **单独运行**: RTF < 0.5（通常）
- **与其他服务一起运行**: RTF < 1.0（目标）
- **首包延迟**: < 0.3秒
- **并发处理能力**: 3-4个并发请求

## 注意事项

1. **多进程模式**: 不建议使用多个worker进程，因为会导致GPU内存问题
2. **内存泄漏**: 定期检查GPU内存使用，确保没有内存泄漏
3. **其他服务影响**: 如果其他服务占用大量GPU资源，可能需要进一步降低GPU内存利用率
4. **网络延迟**: RTF计算不包括网络传输时间，实际用户体验可能受网络影响

## 句子边界优化

### 文本分割优化
- **启用逗号分割**: `comma_split=True`，让文本在逗号和句号处分割
- **调整chunk大小**: `token_max_n=100`, `token_min_n=40`, `merge_len=15`
- **效果**: 每个chunk更可能是一个完整的句子或短语，提升语音自然度

### 流式推理优化
- **句子边界等待机制**: 当达到最小token数时，稍微等待一下，看是否有更多token到达
- **最大等待时间**: 约15ms（3次检查 × 5ms）
- **效果**: 尽量在句子边界处输出chunk，避免在句子中间切断

### token_hop_len配置
- **默认值**: 25
- **可配置**: 通过`--token_hop_len`参数调整
- **建议值**:
  - 追求低延迟: 20-25
  - 平衡: 25-30
  - 追求质量: 30-35

### 使用示例

```bash
# 启用句子边界优化
python tts_api_server.py \
    --model_dir models/Fun-CosyVoice3-0.5B-2512 \
    --load_vllm \
    --fp16 \
    --token_hop_len 30 \
    --gpu_memory_utilization 0.35
```

## 更新日志

- 2024-XX-XX: 添加句子边界优化，启用逗号分割，优化流式推理逻辑
- 2024-XX-XX: 初始版本，包含所有优化措施
