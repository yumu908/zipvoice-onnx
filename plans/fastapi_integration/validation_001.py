# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "websockets",
# ]
# ///
import asyncio
import json
import httpx
import websockets

async def test_non_streaming():
    print("Testing HTTP Non-streaming API...")
    async with httpx.AsyncClient() as client:
        res = await client.post("http://127.0.0.1:7860/api/synthesize", json={
            "text": "测试测试。",
            "language": "zh",
            "ref_wav": "prompt_english_female1.wav",
            "speed": 1.0,
            "num_steps": 4
        }, timeout=30.0)
        assert res.status_code == 200, f"Error: {res.status_code} {res.text}"
        print(f"Success! WAV audio received ({len(res.content)} bytes)\n")

async def test_streaming():
    print("Testing HTTP Streaming (ndjson) API...")
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "http://127.0.0.1:7860/api/synthesize/stream?format=json", json={
            "text": "大漠飞雪，是一场跨越时空的凄美邂逅。",
            "language": "zh",
            "ref_wav": "prompt_english_female1.wav",
            "speed": 1.0,
            "num_steps": 4
        }, timeout=30.0) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    if "error" in data:
                        print("Stream Error:", data["error"])
                    else:
                        print(f"Chunk received: text='{data.get('text')}' index={data.get('index')} audio_length={len(data.get('audio', ''))}")
    print("HTTP Streaming verification finished.\n")

async def test_websocket():
    print("Testing WebSocket API...")
    async with websockets.connect("ws://127.0.0.1:7860/api/ws") as websocket:
        # Send params
        payload = {
            "text": "狂风卷着雪花，在大地上奔走呼号。",
            "language": "zh",
            "ref_wav": "prompt_english_female1.wav",
            "speed": 1.0,
            "num_steps": 4
        }
        await websocket.send(json.dumps(payload))
        
        while True:
            try:
                msg = await websocket.recv()
                data = json.loads(msg)
                event = data.get("event")
                if event == "audio":
                    print(f"WS Audio: text='{data.get('text')}' index={data.get('index')} audio_length={len(data.get('audio', ''))}")
                elif event == "done":
                    print("WS Generation complete.")
                    break
                elif event == "error":
                    print("WS Error:", data.get("message"))
                    break
            except Exception as e:
                print("WS Exception:", e)
                break
    print("WebSocket verification finished.\n")

async def main():
    await test_non_streaming()
    await test_streaming()
    await test_websocket()

if __name__ == "__main__":
    asyncio.run(main())
