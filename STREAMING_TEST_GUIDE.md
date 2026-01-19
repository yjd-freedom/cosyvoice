# 实时流式TTS测试指南

## 概述

`test_streaming_tts.py` 是一个专门用于测试远程TTS服务器流式输出效果的测试工具。它可以：
- 实时接收和显示音频流传输情况
- 测量首包延迟和传输速度
- 支持实时播放音频
- 支持并发测试
- 显示详细的性能统计信息

## 安装依赖

### 基本依赖（已包含在requirements.txt中）
```bash
pip install requests numpy
```

### 实时播放功能（可选）
如果需要实时播放音频，需要安装 `pyaudio`：
```bash
# Windows
pip install pipwin
pipwin install pyaudio

# Linux
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# Mac
brew install portaudio
pip install pyaudio
```

## 基本使用

### 1. 测试本地服务器

```bash
python test_streaming_tts.py \
    --server http://localhost:8000 \
    --text "Hello, this is a streaming test." \
    --output test.wav
```

### 2. 测试远程服务器

```bash
python test_streaming_tts.py \
    --server http://your-company-server.com:8000 \
    --text "This is a test of the remote TTS service." \
    --language en \
    --emotion professional \
    --gender male \
    --output remote_test.wav
```

### 3. 实时播放音频

```bash
python test_streaming_tts.py \
    --server http://your-server.com:8000 \
    --text "Testing real-time audio playback." \
    --play
```

### 4. 测试不同语言和情绪

```bash
# 英语 - 亲切 - 女声
python test_streaming_tts.py \
    --server http://your-server.com:8000 \
    --text "Welcome to our service!" \
    --language en \
    --emotion friendly \
    --gender female

# 俄语 - 专业 - 男声
python test_streaming_tts.py \
    --server http://your-server.com:8000 \
    --text "Добро пожаловать в наш сервис." \
    --language ru \
    --emotion professional \
    --gender male

# 法语 - 兴奋 - 女声
python test_streaming_tts.py \
    --server http://your-server.com:8000 \
    --text "Bienvenue dans notre service!" \
    --language fr \
    --emotion excited \
    --gender female
```

## 性能测试

### 并发测试

测试服务器处理多个并发请求的能力：

```bash
python test_streaming_tts.py \
    --server http://your-server.com:8000 \
    --concurrent 5
```

这将发送5个并发请求，测试服务器的并发处理能力。

### 长文本测试

测试长文本的流式传输效果：

```bash
python test_streaming_tts.py \
    --server http://your-server.com:8000 \
    --text "This is a very long text to test the streaming capability of the TTS system. It should be able to handle long sentences and paragraphs smoothly without interruption. The system should maintain low latency and provide a smooth streaming experience for the end users." \
    --output long_text_test.wav
```

## 输出说明

测试脚本会显示以下信息：

```
============================================================
实时流式TTS测试
============================================================
服务器: http://your-server.com:8000
文本: Hello, this is a test.
语言: en, 情绪: professional, 性别: male
------------------------------------------------------------
正在发送请求...
采样率: 22050 Hz
------------------------------------------------------------
开始接收音频流...

✓ 首包延迟: 0.234 秒

已接收: 10 块, 45.2 KB, 速度: 192.3 KB/s
------------------------------------------------------------
流式传输完成
------------------------------------------------------------
总传输时间: 0.456 秒
首包延迟: 0.234 秒
音频时长: 2.15 秒
传输速度: 198.5 KB/s
数据块数量: 23
总数据量: 94.8 KB
实时因子 (RTF): 0.21x
首包后延迟: 0.222 秒
============================================================
```

### 关键指标说明

- **首包延迟**: 从发送请求到收到第一个音频数据包的时间（越低越好）
- **总传输时间**: 完整音频流传输完成的总时间
- **音频时长**: 实际音频的播放时长
- **实时因子 (RTF)**: 传输时间/音频时长，小于1表示传输速度快于播放速度
- **传输速度**: 音频数据的传输速率

## 性能优化建议

### 1. 网络优化
- 确保服务器和客户端之间的网络连接稳定
- 使用内网连接可以获得最佳性能
- 考虑使用CDN加速（如果部署在公网）

### 2. 服务器优化
- 使用GPU加速推理
- 调整流式输出的chunk大小
- 优化服务器配置

### 3. 客户端优化
- 使用连接池复用连接
- 调整接收缓冲区大小
- 考虑使用WebSocket（如果支持）

## 故障排除

### 连接超时

如果遇到连接超时，检查：
- 服务器地址和端口是否正确
- 防火墙是否允许连接
- 服务器是否正常运行

```bash
# 先测试服务器是否可达
curl http://your-server.com:8000/health
```

### 音频播放问题

如果无法播放音频：
- 检查是否安装了 `pyaudio`
- 检查系统音频设备是否正常
- 可以只保存文件不播放：去掉 `--play` 参数

### 性能问题

如果发现延迟较高：
- 检查网络延迟：`ping your-server.com`
- 检查服务器负载
- 考虑使用更近的服务器节点

## 集成到自动化测试

可以将此脚本集成到CI/CD流程中：

```bash
#!/bin/bash
# 自动化测试脚本

SERVER_URL="http://your-server.com:8000"

# 基本功能测试
python test_streaming_tts.py \
    --server $SERVER_URL \
    --text "Basic functionality test" \
    --output basic_test.wav

# 性能测试
python test_streaming_tts.py \
    --server $SERVER_URL \
    --concurrent 10

# 多语言测试
for lang in en ru fr; do
    python test_streaming_tts.py \
        --server $SERVER_URL \
        --language $lang \
        --text "Test in $lang" \
        --output "test_${lang}.wav"
done
```

## 示例输出

成功的测试应该显示：
- 首包延迟 < 1秒（理想情况下 < 0.5秒）
- RTF < 1（表示传输速度快于播放速度）
- 无错误信息
- 音频文件正常生成

如果看到这些指标，说明流式传输工作正常！
