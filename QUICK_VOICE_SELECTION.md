# 音色选择快速使用指南

## 一、文件准备

确保 `asset` 文件夹下有以下文件：

```
asset/
├── male1.wav    # 孙悟空
├── male2.wav    # 猪八戒
├── male3.wav    # 太乙真人
├── female1.wav  # 武则天
├── female2.wav  # 林志玲
└── female3.wav  # 东北雨姐
```

## 二、音色列表

### 男声音色
- **1**: 孙悟空 - 粗犷沙哑，自信张扬，语速稍快
- **2**: 猪八戒 - 憨厚慵懒，软糯随性，语速慢
- **3**: 太乙真人 - 四川口音，高亮诙谐，语速快嗓门大

### 女声音色
- **1**: 武则天 - 高贵冷艳，雍容华贵，语速缓慢
- **2**: 林志玲 - 甜美娇柔，高柔舒缓，嗲尾音
- **3**: 东北雨姐 - 地道东北方言，豪爽热烈，语速快嗓门大

## 三、API调用示例

### 3.1 使用curl

```bash
# 男声 - 孙悟空（音色1）
curl -X POST "http://localhost:8000/tts" \
  -F "text=你好，这是测试文本。" \
  -F "language=zh" \
  -F "gender=male" \
  -F "voice_id=1" \
  --output output.wav

# 男声 - 猪八戒（音色2）
curl -X POST "http://localhost:8000/tts" \
  -F "text=你好，这是测试文本。" \
  -F "language=zh" \
  -F "gender=male" \
  -F "voice_id=2" \
  --output output.wav

# 女声 - 林志玲（音色2）
curl -X POST "http://localhost:8000/tts" \
  -F "text=你好，这是测试文本。" \
  -F "language=zh" \
  -F "gender=female" \
  -F "voice_id=2" \
  --output output.wav
```

### 3.2 使用Python

```python
import requests

# 查询可用音色
response = requests.get("http://localhost:8000/voices")
voices = response.json()
print("可用音色:", voices)

# 使用指定音色合成
data = {
    "text": "你好，这是测试文本。",
    "language": "zh",
    "gender": "male",
    "voice_id": "1"  # 1=孙悟空, 2=猪八戒, 3=太乙真人
}

response = requests.post("http://localhost:8000/tts", data=data, stream=True)
with open("output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
```

### 3.3 使用测试脚本

```bash
# 测试男声 - 孙悟空（音色1）
python test_streaming_tts.py \
  --server http://localhost:8000 \
  --text "你好，这是测试文本。" \
  --language zh \
  --gender male \
  --voice_id 1 \
  --output test_male1.wav

# 测试女声 - 林志玲（音色2）
python test_streaming_tts.py \
  --server http://localhost:8000 \
  --text "你好，这是测试文本。" \
  --language zh \
  --gender female \
  --voice_id 2 \
  --output test_female2.wav
```

## 四、参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | string | ✅ | - | 要合成的文本 |
| language | string | ❌ | en | 语言：en, ru, fr, zh |
| gender | string | ❌ | male | 性别：male, female |
| voice_id | string | ❌ | 1 | 音色ID：1, 2, 3 |
| stream | boolean | ❌ | true | 是否流式返回 |

## 五、注意事项

1. **文件命名必须准确**：音频文件必须命名为 `male1.wav`, `male2.wav` 等格式
2. **音色ID格式**：使用数字字符串 "1", "2", "3"，不是 "voice1"
3. **向后兼容**：如果不提供 `voice_id`，默认使用音色1
4. **错误处理**：如果指定的音色文件不存在，会自动尝试其他音色
