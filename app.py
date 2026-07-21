# -*- coding: utf-8 -*-
import uvicorn
import os

if __name__ == "__main__":
    # Ensure uploads directory exists
    os.makedirs("./uploads", exist_ok=True)
    
    print("Starting ZipVoice ONNX FastAPI Server on http://127.0.0.1:7860 ...")
    uvicorn.run("zipvoice_onnx.server:app", host="127.0.0.1", port=7860, reload=True)
