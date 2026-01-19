# HTTP/HTTPS API 调用指南

## 服务器地址

- **HTTP**: `http://服务器IP:端口` (例如: `http://localhost:8000`)
- **HTTPS**: `https://服务器IP:端口` (例如: `https://your-domain.com:8443`)

## API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 获取服务信息 |
| `/health` | GET | 健康检查 |
| `/voices` | GET | 获取可用音色列表 |
| `/tts` | POST | 文本转语音（主要接口） |
| `/tts/zero_shot` | POST | 零样本语音合成 |

## 请求参数

### POST /tts

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的文本内容 |
| `language` | string | ❌ | `en` | 语言：`en`(英语)、`ru`(俄语)、`fr`(法语)、`zh`(中文) |
| `emotion` | string | ❌ | `professional` | 情绪：`professional`(专业)、`friendly`(亲切)、`excited`(兴奋) |
| `gender` | string | ❌ | `male` | 性别：`male`(男声)、`female`(女声) |
| `voice_id` | string | ❌ | `1` | 音色ID：`1`, `2`, `3` |
| `enable_emotion` | boolean | ❌ | `None` | 是否启用情绪微调（`None`=自动判断，`true`=启用，`false`=禁用） |
| `sample_rate` | integer | ❌ | `44100` | 输出采样率（Hz），可选值：`16000`, `22050`, `24000`, `44100`, `48000`（默认: `44100`） |
| `stream` | boolean | ❌ | `true` | 是否流式返回 |

#### 音色说明

**男声音色**：
- `1`: 孙悟空（西北方言）
- `2`: 猪八戒
- `3`: 太乙真人（四川方言）

**女声音色**：
- `1`: 武则天
- `2`: 林志玲（台湾口音）
- `3`: 东北雨姐（东北口音）

### POST /tts/zero_shot

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `text` | string | ✅ | 要合成的文本内容 |
| `prompt_text` | string | ✅ | 参考文本（与参考音频对应的文本） |
| `prompt_wav` | string | ✅ | 参考音频文件路径（相对于服务器项目根目录） |
| `language` | string | ❌ | 语言代码（默认：`en`） |
| `emotion` | string | ❌ | 情绪（默认：`professional`） |
| `sample_rate` | integer | ❌ | 输出采样率（Hz），可选值：`16000`, `22050`, `24000`, `44100`, `48000` |
| `stream` | boolean | ❌ | 是否流式返回（默认：`true`） |

### GET /voices

获取所有可用的音色列表及其描述。

**响应示例**：
```json
{
    "male": {
        "1": {
            "name": "孙悟空",
            "description": {
                "zh": "请用四川话表达...",
                "en": "...",
                "ru": "...",
                "fr": "..."
            }
        },
        "2": {...},
        "3": {...}
    },
    "female": {
        "1": {...},
        "2": {...},
        "3": {...}
    }
}
```

## 响应格式

### 成功响应
- **Content-Type**: `audio/wav`
- **响应头**: 
  - `X-Sample-Rate`: 输出音频的采样率（Hz）
  - `Content-Disposition`: `attachment; filename=tts_output.wav`
- **响应体**: WAV格式音频数据（16位PCM，单声道）

### 错误响应
- **Content-Type**: `application/json`
- **状态码**: `400`(参数错误)、`404`(资源未找到)、`500`(服务器错误)、`503`(服务不可用)
- **响应体**: `{"detail": "错误描述信息"}`

## 调用示例

### curl - 基础调用（默认44100Hz）

```bash
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello, this is a test." \
  -F "language=en" \
  -F "gender=male" \
  -F "voice_id=1" \
  --output output.wav
# 注意：默认输出采样率为44100Hz，无需指定sample_rate参数
```

### curl - 指定采样率（44100Hz）

```bash
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello, this is a test." \
  -F "language=en" \
  -F "gender=male" \
  -F "voice_id=1" \
  -F "sample_rate=44100" \
  --output output_44k.wav
```

### curl - 中文调用（带方言，默认44100Hz）

```bash
# 四川方言（太乙真人）- 默认44100Hz
curl -X POST "http://localhost:8000/tts" \
  -F "text=你好，这是测试文本。" \
  -F "language=zh" \
  -F "gender=male" \
  -F "voice_id=3" \
  --output output_sichuan.wav

# 台湾口音（林志玲）- 默认44100Hz
curl -X POST "http://localhost:8000/tts" \
  -F "text=你好，这是测试文本。" \
  -F "language=zh" \
  -F "gender=female" \
  -F "voice_id=2" \
  --output output_taiwan.wav
```

### Python - 基础调用

```python
import requests

url = "http://localhost:8000/tts"
data = {
    "text": "Hello, this is a test.",
    "language": "en",
    "gender": "male",
    "voice_id": "1",
    "stream": True
}

response = requests.post(url, data=data, stream=True)
response.raise_for_status()

# 保存音频文件
with open("output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

# 获取采样率
sample_rate = int(response.headers.get('X-Sample-Rate', 22050))
print(f"音频采样率: {sample_rate} Hz")
```

### Python - 指定采样率（44100Hz）

```python
import requests

url = "http://localhost:8000/tts"
data = {
    "text": "Hello, this is a test.",
    "language": "en",
    "gender": "male",
    "voice_id": "1",
    "sample_rate": 44100,  # 指定输出采样率为44100Hz
    "stream": True
}

response = requests.post(url, data=data, stream=True)
response.raise_for_status()

# 保存音频文件
with open("output_44k.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

# 获取实际采样率
sample_rate = int(response.headers.get('X-Sample-Rate', 22050))
print(f"音频采样率: {sample_rate} Hz")  # 输出: 44100 Hz
```

### Python - 查询可用音色

```python
import requests

# 查询可用音色
response = requests.get("http://localhost:8000/voices")
voices = response.json()

print("男声音色:")
for voice_id, info in voices["male"].items():
    print(f"  {voice_id}: {info['name']}")

print("\n女声音色:")
for voice_id, info in voices["female"].items():
    print(f"  {voice_id}: {info['name']}")
```

### JavaScript - 基础调用

```javascript
async function callTTS(text, language = 'en', gender = 'male', voiceId = '1', sampleRate = null) {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('language', language);
    formData.append('gender', gender);
    formData.append('voice_id', voiceId);
    if (sampleRate) {
        formData.append('sample_rate', sampleRate.toString());
    }
    
    const response = await fetch('http://localhost:8000/tts', {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    // 获取采样率
    const sampleRateHeader = response.headers.get('X-Sample-Rate');
    console.log(`音频采样率: ${sampleRateHeader} Hz`);
    
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
}

// 使用 - 44100Hz采样率
callTTS('Hello, world!', 'en', 'male', '1', 44100);
```

### JavaScript - 查询音色并播放

```javascript
// 查询可用音色
async function getVoices() {
    const response = await fetch('http://localhost:8000/voices');
    const voices = await response.json();
    
    console.log('男声音色:', voices.male);
    console.log('女声音色:', voices.female);
    
    return voices;
}

// 使用指定音色合成
async function synthesizeWithVoice(text, gender, voiceId, sampleRate = 44100) {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('language', 'zh');
    formData.append('gender', gender);
    formData.append('voice_id', voiceId);
    formData.append('sample_rate', sampleRate.toString());
    
    const response = await fetch('http://localhost:8000/tts', {
        method: 'POST',
        body: formData
    });
    
    const audioBlob = await response.blob();
    return audioBlob;
}

// 示例：使用太乙真人（四川方言）合成
getVoices().then(() => {
    synthesizeWithVoice('你好，这是测试文本。', 'male', '3', 44100)
        .then(blob => {
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            audio.play();
        });
});
```

## 采样率说明

### 支持的采样率
- `16000` Hz - 电话质量
- `22050` Hz - 模型默认采样率
- `24000` Hz - 模型默认采样率（部分模型）
- `44100` Hz - CD质量（推荐）
- `48000` Hz - 专业音频质量

### 采样率选择建议
- **44100 Hz**: 适用于大多数应用场景，CD质量，文件大小适中
- **48000 Hz**: 适用于专业音频处理
- **22050/24000 Hz**: 模型原始采样率，无需重采样，处理速度最快
- **16000 Hz**: 适用于电话系统或带宽受限场景

### 注意事项
- 如果未指定 `sample_rate`，将使用模型原始采样率（通常为22050或24000 Hz）
- 指定采样率后，系统会自动对生成的音频进行重采样
- 重采样会增加少量处理时间，但通常可以忽略不计

## 常见错误

| 状态码 | 说明 | 解决方案 |
|--------|------|----------|
| 503 | 模型未加载 | 检查服务器是否正常启动，模型文件是否存在 |
| 400 | 参数错误 | 检查参数值是否正确（语言、情绪、性别、音色ID、采样率） |
| 404 | 资源未找到 | 检查prompt音频文件是否存在，或使用 `/tts/zero_shot` 端点 |
| 500 | 服务器内部错误 | 查看服务器日志，检查服务器资源（内存、GPU等） |

## 注意事项

1. **HTTPS**: 生产环境建议使用HTTPS，需要提供SSL证书
2. **流式处理**: 启用 `stream=true` 可减少延迟，提高用户体验
3. **音频格式**: 返回的音频为WAV格式，16位PCM，单声道
4. **采样率**: 可通过 `sample_rate` 参数指定输出采样率，支持16000、22050、24000、44100、48000 Hz
5. **音色选择**: 使用 `voice_id` 参数选择不同音色，可通过 `/voices` 端点查询可用音色
6. **方言支持**: 部分音色支持方言（四川话、台湾话、东北话），需在中文（`language=zh`）下使用
