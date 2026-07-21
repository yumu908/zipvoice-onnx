# plans/fastapi_integration/validation_001.md

## Objective
Migrate or wrap the ZipVoice ONNX synthesis logic in a FastAPI web service, supporting multiple interfaces (HTTP and WebSocket) and modes (standard non-streaming and chunked sentence streaming).

## Design & Architecture

1. **FastAPI Web App (`src/zipvoice_onnx/server.py`)**
   - Configured with `lifespan` handler to pre-cache models and set up Espeak-NG library paths.
   - Cross-Origin Resource Sharing (CORS) enabled.
   - Built-in static files serving to host the dashboard client.

2. **Endpoints**:
   - `/api/voices` (GET): Discovers system prompts and custom uploaded audio files.
   - `/api/upload` (POST): Accepts file upload of user reference voice WAVs.
   - `/api/synthesize` (POST): Returns a complete WAV audio payload.
   - `/api/synthesize/stream` (POST): Streams sentences sequentially as base64-encoded JSON or raw PCM bytes.
   - `/api/ws` (WebSocket): Real-time duplex communication receiving parameters and sending back chunks.

3. **Premium Web Dashboard UI (`src/zipvoice_onnx/web/`)**:
   - Responsive dark-theme card layout featuring Outfit typography, neon glows, and glassmorphism.
   - Drag-and-drop zone for easy custom voice loading.
   - Interactive sliders for speech speed, inference steps, guidance scale, and tone shift.
   - Dynamic terminal output monitoring engine states.
   - Web Audio API visualizer utilizing an HTML5 Canvas to render live waveforms.

## Validation Execution

We run the validation script using:
```bash
uv run plans/fastapi_integration/validation_001.py
```

### Results Output
```text
Testing HTTP Non-streaming API...
Success! WAV audio received (35308 bytes)

Testing HTTP Streaming (ndjson) API...
Chunk received: text='大漠飞雪，是一场跨越时空的凄美邂逅。' index=0 audio_length=195500
HTTP Streaming verification finished.

Testing WebSocket API...
WS Audio: text='狂风卷着雪花，在大地上奔走呼号。' index=0 audio_length=173656
WS Generation complete.
WebSocket verification finished.
```
All endpoints correctly handle voice configuration parameters, execute model inference, and output valid PCM/WAV formats.
