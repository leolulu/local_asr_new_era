# local_asr_new_era

一个面向 Windows 的本地语音识别工具。项目内置 sherpa-onnx 运行时、SenseVoice 和 ZipFormer 两套模型，支持单个音频/视频文件识别，也支持批量处理文件夹中的媒体文件。

统一入口会根据需要把输入转换为 `16 kHz`、单声道、`16-bit PCM WAV`，然后通过 Silero VAD 分段，再调用指定的 ASR 模型识别。已经符合要求的 WAV 文件会跳过 FFmpeg 转换。

## 功能

- SenseVoice：支持普通话、粤语、英语、日语和韩语，可自动判断语言。
- ZipFormer：支持中文和英语，支持 `greedy_search` 与 `modified_beam_search` 两种解码方式。
- 支持音频、视频和当前目录层级的批量识别；文件夹扫描不会进入子文件夹。
- 支持输出 TXT、SRT、JSON，或同时输出三种格式。
- 支持按文件并发处理；同一个文件选择多个模型时仍按模型顺序依次执行。
- 支持 Ctrl+C 中断时清理正在运行的 FFmpeg、VAD 和 ASR 子进程。
- ZipFormer 遇到已确认的长片段位置表错误时，会自动收紧 VAD 参数并重试，最多 4 次。

## 运行环境

- Windows
- Python 3.10 或更高版本
- [uv](https://docs.astral.sh/uv/)
- FFmpeg：需要加入 `PATH`，或者在命令中通过 `--ffmpeg` 指定完整路径

项目当前以脚本方式运行，Python 代码只使用标准库；模型和 sherpa-onnx 的 Windows 运行文件已放在仓库中，不需要额外安装 Python 推理库。

## 模型和运行文件

默认路径如下：

```text
app/sherpa-onnx/bin/sherpa-onnx-offline.exe
app/sherpa-onnx/bin/sherpa-onnx-vad-with-offline-asr.exe
models/vad/silero_vad.onnx
models/sensevoice/model.int8.onnx
models/sensevoice/tokens.txt
models/zipformer/encoder-epoch-99-avg-1.int8.onnx
models/zipformer/decoder-epoch-99-avg-1.onnx
models/zipformer/joiner-epoch-99-avg-1.int8.onnx
models/zipformer/tokens.txt
```

部分较大的模型文件由 Git LFS 管理。克隆仓库后如果模型文件仍是很小的指针文件，请先安装并配置 Git LFS，再执行：

```powershell
git lfs pull
```

## 快速开始

### 识别单个文件

默认使用 SenseVoice。未指定输出文件选项时，完整识别结果会以 JSON 打印到标准输出：

```powershell
uv run scripts/asr.py "D:\Media\lecture.mp4" --model sensevoice
```

使用 ZipFormer 并在源文件旁生成 SRT 字幕：

```powershell
uv run scripts/asr.py "D:\Media\lecture.m4a" --model zipformer --srt
```

同时生成 TXT、SRT 和 JSON：

```powershell
uv run scripts/asr.py "D:\Media\lecture.mp4" --model sensevoice --all-output
```

### 批量识别文件夹

文件夹模式只处理当前层级中的支持格式，不递归进入子文件夹：

```powershell
uv run scripts/asr.py "D:\Media" --model all --all-output --file-workers 2
```

`--model all` 会按 `sensevoice`、`zipformer` 的顺序运行两套模型，并分别生成例如：

```text
lecture.sensevoice.txt
lecture.sensevoice.srt
lecture.sensevoice.json
lecture.zipformer.txt
lecture.zipformer.srt
lecture.zipformer.json
```

`--file-workers` 默认为 `1`。设置为大于 `1` 时，只增加不同文件之间的并发度，不会改变 ASR 或 VAD 的推理线程数：

```powershell
uv run scripts/asr.py "D:\Media" --model sensevoice --all-output --file-workers 2
```

### 单独转换音频

需要时也可以直接调用音频转换脚本：

```powershell
uv run python scripts/convert_audio.py "D:\Media\lecture.mp3" "D:\Media\lecture.wav" --overwrite
```

默认输出为单声道、16-bit PCM、16 kHz WAV；采样率可通过 `--sample-rate` 修改。

## 输出规则

- 不指定 `--txt`、`--srt`、`--json` 或 `--all-output` 时，识别 JSON 输出到控制台。
- 指定任意输出选项后，结果写入源文件旁边，控制台不再打印识别结果。
- 单模型模式使用源文件名作为基础名，例如 `lecture.txt`、`lecture.srt`、`lecture.json`。
- `--model all` 会为每个模型增加模型名后缀，例如 `lecture.sensevoice.txt`。
- TXT 在 VAD 分段之间保留换行；SRT 使用每个分段的开始和结束时间。
- JSON 包含识别文本和时间分段；控制台 JSON 或文件 JSON 还会包含输入参数、分阶段耗时和资源观测信息。
- 默认情况下，如果本次要生成的任一目标文件已经存在，就跳过对应的模型任务；需要覆盖时使用 `--overwrite`。
- 处理状态和进度写入标准错误流，不会混入 JSON、TXT 或 SRT 内容。

## 常用选项

| 选项 | 说明 |
| --- | --- |
| `--model sensevoice / zipformer / all` | 选择模型，默认 `sensevoice`。 |
| `--file-workers N` | 同时处理的文件数上限，默认 `1`。 |
| `--ffmpeg PATH` | FFmpeg 可执行文件名或完整路径。 |
| `--txt` | 生成 TXT。 |
| `--srt` | 生成带时间轴的 SRT 字幕。 |
| `--json` | 生成 JSON 结果和观测信息。 |
| `--all-output` | 同时生成 TXT、SRT、JSON。 |
| `--overwrite` | 覆盖已有输出文件。 |
| `--language auto / zh / en / ja / ko / yue` | SenseVoice 的语言设置。 |
| `--no-use-itn` | 关闭 SenseVoice 的逆文本规范化。 |
| `--decoding-method greedy_search / modified_beam_search` | ZipFormer 解码方式。 |
| `--num-threads N` | ASR 推理线程数。 |
| `--provider NAME` | ASR 的 ONNX Runtime 执行提供程序，默认 `cpu`。 |
| `--vad-*` | 调整 Silero VAD 阈值、时长、线程数和执行提供程序。 |

完整参数说明：

```powershell
uv run scripts/asr.py --help
```

## 测试

在项目根目录执行：

```powershell
uv run python -m unittest discover -v
```

## 目录结构

```text
app/                   sherpa-onnx Windows 运行文件
models/                VAD、SenseVoice、ZipFormer 模型
scripts/asr.py         统一的文件/文件夹识别入口
scripts/convert_audio.py  独立的音频转 WAV 工具
scripts/sensevoice_asr.py  SenseVoice 调用封装
scripts/zipformer_asr.py   ZipFormer 调用封装
scripts/asr_common.py      公共 VAD、进程和观测逻辑
scripts/process_group.py   子进程生命周期与取消控制
tests/                 单元测试
```
