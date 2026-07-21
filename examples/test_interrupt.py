# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "websockets",
#     "httpx",
# ]
# ///
# -*- coding: utf-8 -*-
"""
ZipVoice ONNX WebSocket Interruption (Barge-in) Demo (Concurrent)

This script demonstrates how to interrupt an ongoing speech synthesis process
and immediately start a new one using two concurrent asyncio tasks:
1. A Receiver Task that continuously listens for incoming audio chunks and events.
2. A Sender Task that sends synthesis and interruption requests independently.
"""

import asyncio
import json
import websockets

WS_URL = "ws://127.0.0.1:7860/api/synthesize"

async def receive_loop(websocket):
    """Continuously receives messages from the WebSocket server in the background."""
    try:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            event = data.get("event")
            
            if event == "audio":
                idx = data.get("index")
                text = data.get("text")
                print(f"[Receiver] Received Audio Chunk {idx+1}: '{text}'")
            elif event == "interrupted":
                print("[Receiver] Server confirmed: Active synthesis interrupted!")
            elif event == "done":
                print("[Receiver] Server confirmed: Synthesis complete.")
            elif event == "error":
                print(f"[Receiver Error] {data.get('message')}")
    except websockets.ConnectionClosed:
        print("[Receiver] Connection closed.")
    except Exception as e:
        print(f"[Receiver Exception] {e}")

async def send_loop(websocket):
    """Sends requests and interruption signals to the server based on application flow."""
    # 1. Send the first long request
    long_text = "这是第一句非常长的话。这是第二句，我们将在这句话合成期间发送打断信号。这是第三句，这一句之后的应该都被打断。"
    print(f"[Sender] Sending request 1: '{long_text}'")
    
    payload_1 = {
      "event": "synthesize",
      "params": {
        "text": long_text,
        "language": "zh",
        "ref_wav": "examples/audio/prompt_english_female1.wav",
        "speed": 1.0,
        "num_steps": 4
      }
    }
    await websocket.send(json.dumps(payload_1))
    
    # 2. Wait for 1.2 seconds to let the server start synthesizing and sending the first chunks
    await asyncio.sleep(1.2)
    
    # 3. Simulate a user "barge-in" (interruption) by sending a new text request
    print("\n[Sender] User interrupts! Sending request 2...")
    new_text = "打断成功！现在开始合成新的这一句紧急播报。"
    payload_2 = {
      "event": "synthesize",
      "params": {
        "text": new_text,
        "language": "zh",
        "ref_wav": "examples/audio/prompt_english_female1.wav",
        "speed": 1.0,
        "num_steps": 4
      }
    }
    await websocket.send(json.dumps(payload_2))
    
    # 4. Wait for the new request to finish generating
    await asyncio.sleep(3.0)
    
    # 5. Send a clean stop/interrupt signal (mute)
    print("\n[Sender] Sending clean mute/interrupt signal...")
    await websocket.send(json.dumps({"event": "interrupt"}))
    await asyncio.sleep(1.0)

async def main():
    print("=== Connecting to WebSocket ===")
    async with websockets.connect(WS_URL) as websocket:
        # Run sender and receiver concurrently
        receiver_task = asyncio.create_task(receive_loop(websocket))
        
        # Run sender task to completion
        await send_loop(websocket)
        
        # Clean up receiver task
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(main())
