# -------------------------------------------------
# check_provider_fixed.py
# -------------------------------------------------
import os, sys, onnxruntime as ort

# 1️⃣ 基础信息
print(ort.get_device())
print("ONNX Runtime version :", ort.__version__)
print("All available providers :", ort.get_available_providers())

# 2️⃣ 选取模型（确保模型文件存在）
MODEL_PATH = os.path.join("model", "fm_decoder.onnx")
if not os.path.isfile(MODEL_PATH):
    sys.exit("[Error] Model not found: " + MODEL_PATH)

# 3️⃣ 构造 Provider 列表：优先尝试 CUDA，若不可用则回退 CPU
preferred_providers = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

# 4️⃣ 创建会话（ONNX Runtime 会自动跳过不可用 Provider）
session = ort.InferenceSession(MODEL_PATH, providers=preferred_providers)

# 5️⃣ 实际生效的 Provider（列表的第一个即为当前使用的）
active = session.get_providers()
print(">>> Active providers (first one is actually used):", active)

# 6️⃣ 给出用户提示
if active[0] == "CUDAExecutionProvider":
    print("[OK] GPU (CUDA) is ACTIVE - inference will run on the GPU.")
else:
    print("[WARN] GPU not available - inference will run on the CPU.")
