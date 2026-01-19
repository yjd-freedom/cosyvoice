# 流式音频API对接指南

## ✅ 流式传输支持确认

**是的，本模块完全支持通过POST方法进行流式音频接收和返回！**

### 技术实现说明

1. **服务器端**：
   - 使用 FastAPI 的 `StreamingResponse` 实现流式响应
   - 模型层支持 `stream=True` 参数，以生成器方式逐步输出音频块
   - 音频数据以16位PCM格式实时传输

2. **客户端**：
   - 使用 `stream=True` 参数接收流式响应
   - 通过 `iter_content()` 逐块接收音频数据
   - 支持实时播放或保存

## 📡 API接口说明

### 端点信息

- **URL**: `POST /tts`
- **Content-Type**: `application/x-www-form-urlencoded` 或 `multipart/form-data`
- **响应类型**: `audio/wav` (流式)

### 请求参数

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的文本内容 |
| `language` | string | ❌ | `en` | 语言代码：`en`(英语)、`ru`(俄语)、`fr`(法语)、`zh`(中文) |
| `emotion` | string | ❌ | `professional` | 情绪：`professional`(专业)、`friendly`(亲切)、`excited`(兴奋) |
| `gender` | string | ❌ | `male` | 性别：`male`(男声)、`female`(女声) |
| `voice_id` | string | ❌ | `1` | 音色ID：`1`、`2`、`3` |
| `stream` | boolean | ❌ | `true` | 是否流式返回（**建议保持为true**） |

### 响应格式

**成功响应**：
- **Content-Type**: `audio/wav`
- **响应头**: 
  - `X-Sample-Rate`: 采样率（通常为22050 Hz）
  - `Content-Disposition`: `attachment; filename=tts_output.wav`
- **响应体**: 流式音频数据（16位PCM，单声道，22050 Hz采样率）

**错误响应**：
- **Content-Type**: `application/json`
- **状态码**: `400`(参数错误)、`500`(服务器错误)、`503`(服务不可用)
- **响应体**: `{"detail": "错误描述信息"}`

## 💻 对接代码示例

### Python 对接示例

#### 基础流式接收

```python
import requests
import wave
import numpy as np

def call_tts_streaming(server_url, text, language="en", emotion="professional", 
                       gender="male", voice_id="1", output_file=None):
    """
    调用TTS API并流式接收音频
    
    Args:
        server_url: 服务器地址，如 "http://localhost:8000"
        text: 要合成的文本
        language: 语言代码
        emotion: 情绪
        gender: 性别
        voice_id: 音色ID
        output_file: 输出文件路径（可选）
    
    Returns:
        tuple: (音频数据bytes, 采样率)
    """
    url = f"{server_url}/tts"
    
    # 准备表单数据
    data = {
        "text": text,
        "language": language,
        "emotion": emotion,
        "gender": gender,
        "voice_id": voice_id,
        "stream": True  # 启用流式传输
    }
    
    # 发送POST请求，启用流式接收
    response = requests.post(url, data=data, stream=True, timeout=300)
    response.raise_for_status()
    
    # 获取采样率
    sample_rate = int(response.headers.get('X-Sample-Rate', 22050))
    
    # 流式接收音频数据
    audio_chunks = []
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            audio_chunks.append(chunk)
            # 可以在这里实时处理音频块
    
    # 合并所有音频块
    audio_data = b''.join(audio_chunks)
    
    # 保存为WAV文件（如果需要）
    if output_file:
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位 = 2字节
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        print(f"音频已保存到: {output_file}")
    
    return audio_data, sample_rate

# 使用示例
if __name__ == "__main__":
    audio_data, sample_rate = call_tts_streaming(
        server_url="http://localhost:8000",
        text="Hello, this is a streaming test.",
        language="en",
        emotion="professional",
        gender="male",
        voice_id="1",
        output_file="output.wav"
    )
    print(f"采样率: {sample_rate} Hz")
    print(f"音频数据大小: {len(audio_data)} 字节")
```

#### 实时播放流式音频

```python
import requests
import pyaudio
import numpy as np

def stream_tts_with_playback(server_url, text, language="en", 
                             emotion="professional", gender="male", voice_id="1"):
    """
    流式接收并实时播放音频
    """
    url = f"{server_url}/tts"
    data = {
        "text": text,
        "language": language,
        "emotion": emotion,
        "gender": gender,
        "voice_id": voice_id,
        "stream": True
    }
    
    response = requests.post(url, data=data, stream=True, timeout=300)
    response.raise_for_status()
    
    sample_rate = int(response.headers.get('X-Sample-Rate', 22050))
    
    # 初始化音频播放器
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        output=True
    )
    
    try:
        # 流式接收并实时播放
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# 使用示例
stream_tts_with_playback(
    server_url="http://localhost:8000",
    text="This is a real-time streaming audio test."
)
```

#### 带进度监控的流式接收

```python
import requests
import time

def stream_tts_with_progress(server_url, text, **kwargs):
    """
    流式接收音频并显示进度
    """
    url = f"{server_url}/tts"
    data = {
        "text": text,
        "stream": True,
        **kwargs
    }
    
    start_time = time.time()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0
    
    response = requests.post(url, data=data, stream=True, timeout=300)
    response.raise_for_status()
    
    sample_rate = int(response.headers.get('X-Sample-Rate', 22050))
    audio_chunks = []
    
    print("开始接收音频流...")
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            chunk_time = time.time()
            
            # 记录首包延迟
            if first_chunk_time is None:
                first_chunk_time = chunk_time
                first_chunk_latency = first_chunk_time - start_time
                print(f"✓ 首包延迟: {first_chunk_latency:.3f} 秒")
            
            total_bytes += len(chunk)
            chunk_count += 1
            audio_chunks.append(chunk)
            
            # 显示进度
            if chunk_count % 10 == 0:
                elapsed = chunk_time - start_time
                if elapsed > 0:
                    speed = total_bytes / elapsed / 1024  # KB/s
                    print(f"已接收: {chunk_count} 块, {total_bytes/1024:.1f} KB, "
                          f"速度: {speed:.1f} KB/s", end='\r')
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    audio_data = b''.join(audio_chunks)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_duration = len(audio_array) / sample_rate
    
    print(f"\n传输完成:")
    print(f"  总传输时间: {total_duration:.3f} 秒")
    print(f"  首包延迟: {first_chunk_latency:.3f} 秒")
    print(f"  音频时长: {audio_duration:.2f} 秒")
    print(f"  实时因子 (RTF): {total_duration / audio_duration:.2f}x")
    
    return audio_data, sample_rate
```

### JavaScript/Node.js 对接示例

#### 基础流式接收

```javascript
const axios = require('axios');
const fs = require('fs');

async function callTTSStreaming(serverUrl, text, options = {}) {
    const {
        language = 'en',
        emotion = 'professional',
        gender = 'male',
        voiceId = '1',
        outputFile = null
    } = options;
    
    const formData = new FormData();
    formData.append('text', text);
    formData.append('language', language);
    formData.append('emotion', emotion);
    formData.append('gender', gender);
    formData.append('voice_id', voiceId);
    formData.append('stream', 'true');
    
    try {
        const response = await axios.post(`${serverUrl}/tts`, formData, {
            responseType: 'stream',
            headers: formData.getHeaders()
        });
        
        const sampleRate = parseInt(response.headers['x-sample-rate'] || '22050');
        
        if (outputFile) {
            const writer = fs.createWriteStream(outputFile);
            response.data.pipe(writer);
            
            return new Promise((resolve, reject) => {
                writer.on('finish', () => {
                    resolve({ sampleRate, file: outputFile });
                });
                writer.on('error', reject);
            });
        } else {
            // 收集所有数据块
            const chunks = [];
            response.data.on('data', (chunk) => {
                chunks.push(chunk);
            });
            
            return new Promise((resolve, reject) => {
                response.data.on('end', () => {
                    const audioData = Buffer.concat(chunks);
                    resolve({ audioData, sampleRate });
                });
                response.data.on('error', reject);
            });
        }
    } catch (error) {
        console.error('TTS请求失败:', error.message);
        throw error;
    }
}

// 使用示例
callTTSStreaming('http://localhost:8000', 'Hello, world!', {
    language: 'en',
    emotion: 'friendly',
    gender: 'female',
    outputFile: 'output.wav'
}).then(result => {
    console.log('音频已保存，采样率:', result.sampleRate);
});
```

#### 浏览器端流式接收

```javascript
async function callTTSInBrowser(serverUrl, text, options = {}) {
    const {
        language = 'en',
        emotion = 'professional',
        gender = 'male',
        voiceId = '1',
        onChunk = null  // 回调函数，接收每个音频块
    } = options;
    
    const formData = new FormData();
    formData.append('text', text);
    formData.append('language', language);
    formData.append('emotion', emotion);
    formData.append('gender', gender);
    formData.append('voice_id', voiceId);
    formData.append('stream', 'true');
    
    const response = await fetch(`${serverUrl}/tts`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const sampleRate = parseInt(response.headers.get('X-Sample-Rate') || '22050');
    const reader = response.body.getReader();
    const chunks = [];
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        chunks.push(value);
        
        // 如果提供了回调函数，实时处理音频块
        if (onChunk) {
            onChunk(value);
        }
    }
    
    // 合并所有块
    const audioData = new Uint8Array(
        chunks.reduce((acc, chunk) => acc + chunk.length, 0)
    );
    let offset = 0;
    for (const chunk of chunks) {
        audioData.set(chunk, offset);
        offset += chunk.length;
    }
    
    return { audioData, sampleRate };
}

// 使用示例
callTTSInBrowser('http://localhost:8000', 'Hello, world!', {
    language: 'en',
    onChunk: (chunk) => {
        console.log('收到音频块:', chunk.length, '字节');
        // 可以在这里实时播放或处理音频块
    }
}).then(result => {
    console.log('音频接收完成，采样率:', result.sampleRate);
    // 创建音频对象并播放
    const audioBlob = new Blob([result.audioData], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
});
```

### curl 命令行示例

```bash
# 基础流式请求
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello, this is a streaming test." \
  -F "language=en" \
  -F "emotion=professional" \
  -F "gender=male" \
  -F "voice_id=1" \
  -F "stream=true" \
  --output output.wav

# 显示进度
curl -X POST "http://localhost:8000/tts" \
  -F "text=This is a long text to test streaming capabilities." \
  -F "language=en" \
  -F "stream=true" \
  --progress-bar \
  --output output.wav
```

## 🔧 对接要点

### 1. 必须使用 `stream=True`

在请求参数中设置 `stream=True`（或表单中的 `"stream": "true"`），这是启用流式传输的关键。

### 2. 客户端必须启用流式接收

- **Python requests**: 使用 `stream=True` 参数
- **JavaScript fetch**: 使用 `response.body.getReader()` 读取流
- **curl**: 默认支持流式接收

### 3. 音频格式说明

- **格式**: 16位PCM，单声道
- **采样率**: 22050 Hz（从响应头 `X-Sample-Rate` 获取）
- **数据格式**: 原始PCM字节流，不是标准WAV文件格式
- **保存为WAV**: 需要添加WAV文件头，或使用示例代码中的方法

### 4. 错误处理

```python
try:
    response = requests.post(url, data=data, stream=True, timeout=300)
    response.raise_for_status()
    
    # 处理流式数据
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            # 处理音频块
            pass
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 503:
        print("服务不可用，模型未加载")
    elif e.response.status_code == 400:
        error_detail = e.response.json()
        print(f"参数错误: {error_detail.get('detail')}")
    else:
        print(f"HTTP错误: {e}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
```

### 5. 性能优化建议

1. **调整chunk_size**: 根据网络情况调整接收块大小（建议4096-8192字节）
2. **超时设置**: 设置合理的超时时间（建议300秒）
3. **连接复用**: 使用连接池复用HTTP连接
4. **并发控制**: 根据服务器性能控制并发请求数量

## 📊 性能指标

### 典型性能表现

- **首包延迟**: < 1秒（理想情况下 < 0.5秒）
- **实时因子 (RTF)**: < 1（表示传输速度快于播放速度）
- **传输速度**: 取决于网络和服务器性能，通常 > 100 KB/s

### 测试工具

项目提供了专门的流式测试工具：

```bash
python test_streaming_tts.py \
    --server http://your-server.com:8000 \
    --text "Test streaming TTS" \
    --language en \
    --output test.wav
```

## ⚠️ 注意事项

1. **音频格式**: 返回的是原始PCM数据，不是标准WAV文件。需要添加WAV头或使用提供的保存方法。

2. **流式传输优势**: 
   - 降低首包延迟
   - 支持实时播放
   - 减少内存占用

3. **网络要求**: 
   - 确保网络连接稳定
   - 内网连接可获得最佳性能
   - 公网部署建议使用HTTPS

4. **服务器资源**: 
   - 流式传输会占用服务器资源直到传输完成
   - 建议设置合理的超时时间
   - 监控服务器负载

## 🔗 相关文档

- [API调用指南](./API_CALLING_GUIDE.md) - 基础API使用说明
- [流式测试指南](./STREAMING_TEST_GUIDE.md) - 流式传输测试方法
- [音色选择指南](./VOICE_SELECTION_GUIDE.md) - 音色配置说明

## 📞 技术支持

如有问题，请检查：
1. 服务器是否正常运行（访问 `/health` 端点）
2. 参数是否正确（语言、情绪、性别等）
3. 网络连接是否正常
4. 服务器日志中的错误信息
