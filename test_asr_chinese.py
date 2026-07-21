# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
# ]
# ///
import httpx
res = httpx.post(
    "http://127.0.0.1:7860/api/synthesize",
    json={
        "text": "测试中文自动转录参考音频功能是否正常。",
        "language": "zh",
        "ref_wav": "audio_chinese.wav",
        "speed": 1.0,
        "num_steps": 4
    },
    timeout=60.0
)
print("Status:", res.status_code)
print("Response size:", len(res.content))
