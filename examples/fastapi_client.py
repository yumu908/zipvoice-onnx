# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "websockets",
# ]
# ///
# -*- coding: utf-8 -*-
"""
ZipVoice ONNX FastAPI Client Example

This script demonstrates how to interact with the ZipVoice ONNX FastAPI server programmatically.
It covers:
1. Listing available reference voices
2. Uploading a custom reference voice
3. Generating a full WAV audio file (HTTP Non-streaming)
4. Streaming sentence-by-sentence audio (HTTP Streaming)
5. Establishing a real-time duplex stream (WebSocket)
"""

import asyncio
import base64
import json
import os
import httpx
import websockets

SERVER_URL = "http://127.0.0.1:7860"
WS_URL = "ws://127.0.0.1:7860"

async def list_voices():
    print("=== 1. Listing Available Voices ===")
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{SERVER_URL}/api/voices")
        if res.status_code == 200:
            voices = res.json()
            for v in voices:
                print(f"- Name: {v['name']} (Path: {v['path']}, Type: {v['type']})")
        else:
            print("Failed to list voices:", res.text)
    print()

async def generate_non_streaming():
    print("=== 2. HTTP Non-Streaming (Full WAV Generation) ===")
    payload = {
        "text": "测试中文生成。这是非流式接口测试。",
        "language": "zh",
        "ref_wav": "examples/audio/prompt_english_female1.wav",
        "speed": 1.0,
        "num_steps": 4
    }
    
    async with httpx.AsyncClient() as client:
        # Use HTTP POST /api/synthesize for non-streaming WAV generation
        res = await client.post(f"{SERVER_URL}/api/synthesize", json=payload, timeout=60.0)
        if res.status_code == 200:
            out_file = "output_non_streaming.wav"
            with open(out_file, "wb") as f:
                f.write(res.content)
            print(f"Success! Saved synthesized audio to: {out_file} ({len(res.content)} bytes)")
        else:
            print("Failed non-streaming generation:", res.text)
    print()

async def generate_streaming_http():
    print("=== 3. HTTP Streaming (Sentence-by-Sentence Chunks) ===")
    payload = {
        "text": "大漠飞雪，是一场跨越时空的凄美邂逅。狂风卷着雪花，在大地上奔走呼号。",
        "language": "zh",
        "ref_wav": "examples/audio/prompt_english_female1.wav",
        "speed": 1.0,
        "num_steps": 4
    }
    
    async with httpx.AsyncClient() as client:
        # Use HTTP POST /api/generate_stream for sentence-by-sentence HTTP streaming
        async with client.stream(
            "POST", 
            f"{SERVER_URL}/api/generate_stream?format=json", 
            json=payload, 
            timeout=60.0
        ) as response:
            if response.status_code != 200:
                print("Failed to establish stream:", response.status_code)
                return
                
            async for line in response.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    if "error" in data:
                        print("Stream Error:", data["error"])
                    else:
                        idx = data.get("index")
                        text = data.get("text")
                        audio_b64 = data.get("audio", "")
                        print(f"Received chunk {idx + 1} for sentence: '{text}' (Audio size: {len(audio_b64)} chars)")
    print()

async def generate_streaming_websocket():
    print("=== 4. WebSocket Duplex Streaming ===")
    payload = {
        "text": "狂风卷着雪花，在大地上奔走呼号。大自然在怒吼，宣泄着无穷的力量。",
        "language": "zh",
        "ref_wav": "examples/audio/prompt_english_female1.wav",
        "speed": 1.0,
        "num_steps": 4
    }
    
    # Use WebSocket ws://.../api/synthesize for duplex streaming
    async with websockets.connect(f"{WS_URL}/api/synthesize") as websocket:
        print("Connected to WebSocket. Sending parameters...")
        await websocket.send(json.dumps(payload))
        
        while True:
            try:
                msg = await websocket.recv()
                data = json.loads(msg)
                event = data.get("event")
                
                if event == "audio":
                    idx = data.get("index")
                    text = data.get("text")
                    audio_b64 = data.get("audio", "")
                    print(f"WS Chunk {idx + 1} received: '{text}' (Audio size: {len(audio_b64)} chars)")
                elif event == "done":
                    print("WS Generation complete.")
                    break
                elif event == "error":
                    print("WS Error Event:", data.get("message"))
                    break
            except Exception as e:
                print("WS Connection exception:", e)
                break
    print()

async def main():
    await list_voices()
    await generate_non_streaming()
    await generate_streaming_http()
    await generate_streaming_websocket()

if __name__ == "__main__":
    asyncio.run(main())
