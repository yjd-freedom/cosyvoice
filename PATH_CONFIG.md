# 路径配置说明

## 文件夹结构

项目使用以下文件夹结构：

```
CosyVoice-main/
├── models/                    # 模型文件夹
│   └── Fun-CosyVoice3-0.5B-2512/  # 模型文件
│       ├── cosyvoice3.yaml
│       ├── llm.pt
│       ├── flow.pt
│       ├── hift.pt
│       └── ...
├── asset/                     # 参考音频文件夹
│   ├── en_male.wav           # 英语男声参考音频
│   ├── en_female.wav          # 英语女声参考音频
│   ├── ru_male.wav            # 俄语男声参考音频
│   ├── ru_female.wav           # 俄语女声参考音频
│   ├── fr_male.wav             # 法语男声参考音频
│   └── fr_female.wav           # 法语女声参考音频
└── tts_api_server.py          # API服务器
```

## 自动路径选择

### 模型路径

系统会按以下优先级查找模型：

1. 命令行参数 `--model_dir`
2. 环境变量 `MODEL_DIR`
3. 默认路径 `models/Fun-CosyVoice3-0.5B-2512`

### Prompt音频路径

系统会根据请求的语言和性别自动选择对应的prompt文件：

- **英语 + 男声** → `asset/en_male.wav`
- **英语 + 女声** → `asset/en_female.wav`
- **俄语 + 男声** → `asset/ru_male.wav`
- **俄语 + 女声** → `asset/ru_female.wav`
- **法语 + 男声** → `asset/fr_male.wav`
- **法语 + 女声** → `asset/fr_female.wav`

如果找不到精确匹配的文件，系统会：
1. 尝试使用同语言的其他性别文件
2. 尝试使用通用prompt文件（如果存在）

## 配置示例

### 使用默认路径

```bash
# 直接启动，使用默认路径
python tts_api_server.py
```

### 使用自定义模型路径

```bash
# 方式1: 命令行参数
python tts_api_server.py --model_dir /path/to/your/model

# 方式2: 环境变量
export MODEL_DIR=/path/to/your/model
python tts_api_server.py
```

### 使用启动脚本

```bash
# Linux/Mac
export MODEL_DIR=models/Fun-CosyVoice3-0.5B-2512
./start_tts_server.sh

# Windows
set MODEL_DIR=models\Fun-CosyVoice3-0.5B-2512
start_tts_server.bat
```

## 注意事项

1. **模型路径**: 必须指向包含 `cosyvoice3.yaml`（或 `cosyvoice2.yaml`）的模型目录
2. **Prompt文件**: 建议为每种语言和性别组合准备对应的音频文件，以获得最佳效果
3. **文件格式**: Prompt音频文件应为WAV格式，采样率16000Hz或22050Hz
