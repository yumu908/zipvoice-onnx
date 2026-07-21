<img width="1831" height="1707" alt="image" src="https://github.com/user-attachments/assets/212f330f-f519-418b-929a-392ce5bea797" />

# ZipVoice ONNX

ZipVoice ONNX is a library for performing zero-shot voice cloning and text-to-speech (TTS) synthesis using the ZipVoice model.

See ZipVoice on [Hugging Face](https://huggingface.co/k2-fsa/ZipVoice) and [GitHub](https://github.com/k2-fsa/ZipVoice).

---

## Installation

This project utilizes the `uv` toolchain for dependency and python environment management.

```console
# Sync dependencies
uv pip install -r requirements.txt
```

---

## Usage

You can run direct inference scripts using the files in the `examples/` directory:

```console
# Run English synthesis example
uv run examples/english.py

# Run Chinese synthesis example
uv run examples/chinese.py
```

---

=======

## FastAPI Server & Web Studio

The project includes a FastAPI-powered speech synthesis server and a premium Web Studio dashboard for interactive testing of zero-shot voice cloning.

### Running the Server

To start the local development server:

```console
uv run python run_server.py
```

Once started, the following interfaces are available:

* **Web Studio Dashboard**: [http://127.0.0.1:7860](http://127.0.0.1:7860)
* **Interactive Swagger API Docs**: [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs)
  * *Purpose*: Built for live interactive testing. It features a "Try it out" button on each endpoint so you can send real requests and see instant responses directly in the browser.
  * *UI Button*: Also accessible via the **API Docs** button in the Web Studio header.
* **ReDoc API Documentation**: [http://127.0.0.1:7860/redoc](http://127.0.0.1:7860/redoc)
  * *Purpose*: Optimized for structured reading and clean navigation. Ideal for sharing reference materials and browsing the API layout without executing requests.

---

## API Reference & Endpoints

All endpoints support optional or auto-transcribed reference text.

### Speech Synthesis Routes

#### 1. HTTP 普通合成 (非流式)

生成完整的音频并以 WAV 文件形式返回。

* **路径**: `POST /api/synthesize`
* **Payload**:

  ```json
  {
    "text": "目标生成文本。",
    "language": "zh",
    "ref_wav": "examples/audio/prompt_english_female1.wav",
    "ref_text": null,
    "speed": 1.0,
    "num_steps": 4
  }
  ```

#### 2. HTTP 分段流式合成

按句子切割目标文本，并实时流式返回音频数据块。

* **路径**: `POST /api/generate_stream`
* **查询参数**: `format=json` (按行返回 ndjson 格式的 JSON 字符串，包含 Base64 音频和元数据) 或 `format=pcm` (直接流式返回原始 16-bit 24kHz Mono 脉冲编码调制二进制 PCM 字节)

#### 3. WebSocket 双工流式合成（支持实时打断 / Barge-in）

建立持久的 WebSocket 双向长连接，可在单个连接内连续发送多次合成请求，并支持**实时打断（Barge-in）**。

* **路径**: `WebSocket /api/synthesize`

##### 协议事件规范 (Events Protocol)

###### A. 客户端发送的事件 (Client -> Server)

* **`synthesize` (语音合成)**
  发送 JSON 请求开启合成。若当前有正在播放/合成的任务，会自动被新请求打断。

  ```json
  {
    "event": "synthesize",
    "params": {
      "text": "待合成文本",
      "language": "zh",
      "ref_wav": "examples/audio/prompt.wav",
      "speed": 1.0,
      "num_steps": 4
    }
  }
  ```

  *(注：亦可直接发送纯参数对象如 `{"text": "..."}`，服务端会默认识别为 `synthesize` 事件)*
* **`interrupt` (仅打断)**
  立即中止当前正在进行的合成，不启动新合成。

  ```json
  {
    "event": "interrupt"
  }
  ```

###### B. 服务端发送的事件 (Server -> Client)

* **`audio` (音频推送)**
  每完成一句话的合成便向客户端推送音频块。

  ```json
  {
    "event": "audio",
    "index": 0,            // 句子索引 (从 0 开始)
    "text": "分句的文本",
    "audio": "UklGRi...",  // Base64 编码的 16-bit 24kHz Mono 原始 PCM 字节
    "sample_rate": 24000,
    "done": false          // 是否是本段文本的最后一句话
  }
  ```

* **`interrupted` (打断确认)**
  确认之前的合成任务已被中止。

  ```json
  {
    "event": "interrupted"
  }
  ```

* **`done` (完成)**
  当前合成任务已全部完成并发送完毕。

  ```json
  {
    "event": "done"
  }
  ```

* **`error` (错误)**
  推理或文件读取失败。

  ```json
  {
    "event": "error",
    "message": "错误原因"
  }
  ```

### Utility & Transcription Routes

* **List Voices** (`GET /api/voices`): Lists all system and uploaded reference WAV files.
* **Get Voice Audio** (`GET /api/voices/audio?path=<voice_path>`): Serves the raw `.wav` audio for selected reference voice previewing.
* **Auto Transcribe** (`POST /api/voices/transcribe`): Triggers Whisper to transcribe the reference voice audio.

### Programmatic Client Examples

#### 基础客户端测试

运行如下脚本来测试三种基本合成模式（HTTP 非流式、HTTP 流式、WebSocket 基础流式）：

```console
uv run .\examples\fastapi_client.py
```

#### 实时打断测试

运行如下脚本来测试 WebSocket 实时打断（Barge-in）功能：

```console
uv run .\examples\test_interrupt.py
```

---

## Code Obfuscation & Bundling (代码混淆与打包)

本项目提供了一个自动化的代码混淆打包工具 `build.py`。它使用 **Pyarmor** 对 `src/zipvoice_onnx` 的核心 Python 代码进行混淆保护，将二进制依赖运行时嵌入包内，拷贝前端静态文件，并生成独立的启动脚本。

### 打包步骤

1. **执行打包脚本**：
   在项目根目录下运行以下命令：

   ```console
   uv run python build.py
   ```

2. **打包产物**：
   打包完成后，所有包产物都输出到项目根目录下的 `dist/` 目录中：
   * `dist/zipvoice_onnx/`：已混淆的 Python 核心包（包含嵌入其中的 `pyarmor_runtime_000000` 支持模块）及静态 Web UI 资源。
   * `dist/run_server.py`：启动混淆后服务器的入口脚本（配置为单进程生产模式，默认监听 `7860` 端口）。

### 运行混淆后的服务器

直接在项目根目录下执行：

```console
uv run python dist/run_server.py
```

此时，服务器将正常启动运行在端口 `7860` 上，所有的核心推理、分词和路由逻辑均已受到混淆保护，不易被反编译或阅读。

---

## Important Development Details & Fixes

* **Whisper CPU FP16 Warn Fix**: Fixed the PyTorch FP16 warning on CPU execution (`FP16 is not supported on CPU; using FP32 instead`) by checking device capability dynamically (`fp16=is_cuda`) during ASR transcription requests.
* **Reference Audio Preview Isolation**: Corrected the play/pause logic for the reference voice player so that pausing keeps the playhead state, allowing seamless resumption without resetting to `0`.
* **Espeak-NG Cache Integration**: The lifespan startup script dynamically downloads and caches espeak-ng dictionaries in `./model-en-distilled` or local cache path, preventing OS dependencies configuration issues.

>>>>>>> d794748 (修改ui，测试运行)
