# 快速开始指南

## 第一步：准备模型

### 下载模型（推荐使用Fun-CosyVoice3-0.5B）

**国内用户（使用ModelScope）**:
```python
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='models/Fun-CosyVoice3-0.5B-2512')
```

**海外用户（使用HuggingFace）**:
```python
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='models/Fun-CosyVoice3-0.5B-2512')
```

或者直接使用模型ID（会自动下载）:
```bash
python tts_api_server.py --model_dir FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

## 第二步：启动服务器

### Windows系统

```cmd
# 方式1: 使用启动脚本
start_tts_server.bat

# 方式2: 直接运行
python tts_api_server.py --model_dir models/Fun-CosyVoice3-0.5B-2512 --port 8000
```

### Linux/Mac系统

```bash
# 方式1: 使用启动脚本
chmod +x start_tts_server.sh
./start_tts_server.sh

# 方式2: 直接运行
python tts_api_server.py --model_dir models/Fun-CosyVoice3-0.5B-2512 --port 8000
```

### 使用环境变量

```bash
export MODEL_DIR=models/Fun-CosyVoice3-0.5B-2512
export PORT=8000
python tts_api_server.py
```

## 第三步：测试API

### 方法1: 使用测试脚本

```bash
python test_tts_api.py http://localhost:8000
```

### 方法2: 使用curl

```bash
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello, this is a test." \
  -F "language=en" \
  -F "emotion=professional" \
  -F "gender=male" \
  --output test.wav
```

### 方法3: 使用Python客户端

```bash
python tts_client_example.py \
    --server http://localhost:8000 \
    --text "Hello, world!" \
    --language en \
    --emotion friendly \
    --gender female \
    --output output.wav
```

## 第四步：集成到您的项目

### Python集成示例

```python
import requests

def tts_synthesize(text, language="en", emotion="professional", gender="male"):
    url = "http://localhost:8000/tts"
    data = {
        "text": text,
        "language": language,
        "emotion": emotion,
        "gender": gender,
        "stream": True
    }
    
    response = requests.post(url, data=data, stream=True)
    response.raise_for_status()
    
    audio_data = b''
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            audio_data += chunk
    
    return audio_data

# 使用
audio = tts_synthesize(
    text="Welcome to our service!",
    language="en",
    emotion="friendly",
    gender="female"
)

with open("output.wav", "wb") as f:
    f.write(audio)
```

## 常见问题

### Q: 模型加载失败？

A: 确保：
1. 模型文件已完整下载
2. 有足够的磁盘空间和内存
3. 检查日志中的详细错误信息

### Q: 找不到prompt文件？

A: 确保 `asset/zero_shot_prompt.wav` 存在，或使用 `/tts/zero_shot` 端点提供自定义prompt。

### Q: 如何启用HTTPS？

A: 使用SSL证书和密钥：
```bash
python tts_api_server.py \
    --model_dir models/Fun-CosyVoice3-0.5B-2512 \
    --ssl_keyfile /path/to/key.pem \
    --ssl_certfile /path/to/cert.pem \
    --port 8443
```

### Q: 性能优化建议？

A: 
1. 使用GPU加速（需要CUDA）
2. 使用流式输出减少延迟
3. 在生产环境中使用连接池
4. 考虑使用TensorRT加速（需要额外配置）

## 下一步

- 查看 [TTS_API_README.md](TTS_API_README.md) 了解详细的API文档
- 查看 [test_tts_api.py](test_tts_api.py) 了解更多使用示例
- 查看 CosyVoice 官方文档了解更多高级功能
