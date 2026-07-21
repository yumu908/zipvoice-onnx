# -*- coding: utf-8 -*-
"""
ZipVoice ONNX Build & Packaging Script

This script automates:
1. Cleaning and preparing the 'dist' and 'build' directories.
2. Obfuscating the 'src/zipvoice_onnx' package using Pyarmor.
3. Generating a launcher script with explicit imports so PyInstaller can trace the dependencies of the obfuscated package.
4. Bundling the application into a standalone folder with a .exe launcher using PyInstaller.
5. Copying necessary models and reference audio so that the executable works out-of-the-box.
"""

import os
import shutil
import subprocess
import sys

def main():
    print("=== Starting ZipVoice ONNX Build & Package Process ===")
    
    # 1. Clean existing dist and build directories
    dist_dir = os.path.abspath("dist")
    build_dir = os.path.abspath("build")
    
    # 0. Check if the output folder is locked by another process (e.g. user terminal inside dist/run_server)
    exe_dist_dir = os.path.join(dist_dir, "run_server")
    if os.path.exists(exe_dist_dir):
        try:
            test_path = exe_dist_dir + "_test_lock"
            os.rename(exe_dist_dir, test_path)
            os.rename(test_path, exe_dist_dir)
        except OSError:
            print("\n" + "="*80)
            print("[ERROR] Permission Denied: The directory 'dist/run_server' is locked by another process.")
            print("This usually happens if:")
            print("1. You have a Command Prompt (cmd.exe), PowerShell, or File Explorer open inside 'dist/run_server'.")
            print("   -> Please close it, run 'cd ..' to exit that directory, or navigate back to the root.")
            print("2. A previous instance of the server (run_server.exe) is still running.")
            print("   -> Please close the running server.")
            print("="*80 + "\n")
            sys.exit(1)
            
    print("\nCleaning existing build/dist directories...")
    for path in [dist_dir, build_dir]:
        if os.path.exists(path):
            print(f"Removing {path}...")
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Warning: Could not delete {path} ({e}). Retrying by deleting files individually...")
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception:
                            pass
                            
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)
    
    # 2. Run pyarmor gen
    print("\nRunning Pyarmor code obfuscation...")
    cmd_pyarmor = [
        sys.executable, "-m", "pyarmor.cli", "gen",
        "-i",                 # Nest runtime package inside zipvoice_onnx package
        "-O", "dist",         # Output path
        "-r",                 # Search scripts recursively
        "src/zipvoice_onnx"   # Target scripts/package folder
    ]
    
    print(f"Executing command: {' '.join(cmd_pyarmor)}")
    try:
        subprocess.run(cmd_pyarmor, check=True)
        print("Pyarmor obfuscation completed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n[Error] Pyarmor obfuscation failed!", file=sys.stderr)
        sys.exit(1)
        
    # 3. Generate launcher run_server.py in dist/ with static imports for PyInstaller
    launcher_path = os.path.join(dist_dir, "app.py")
    print(f"\nCreating launcher script with dependency tracing: {launcher_path}")
    
    launcher_content = """# -*- coding: utf-8 -*-
# Explicit imports for PyInstaller static dependency analysis
import numpy
import onnxruntime
import fastapi
import fastapi.staticfiles
import fastapi.middleware.cors
import fastapi.responses
import uvicorn
import soundfile
import websockets
import pydantic
import espeakng_loader
import phonemizer
import phonemizer.backend.espeak.wrapper
import pypinyin
import librosa
import whisper
import torch
import tqdm

import os
import sys

# Ensure the parent directory of zipvoice_onnx is in the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports of all submodules so PyInstaller bundles them
import zipvoice_onnx
import zipvoice_onnx.audio
import zipvoice_onnx.model
import zipvoice_onnx.tokenizer
import zipvoice_onnx.vocoder
import zipvoice_onnx.server

if __name__ == "__main__":
    # Ensure uploads directory exists
    os.makedirs("./uploads", exist_ok=True)
    
    print("Starting Obfuscated ZipVoice ONNX FastAPI Server on http://127.0.0.1:7860 ...")
    uvicorn.run("zipvoice_onnx.server:app", host="127.0.0.1", port=7860, reload=False)
"""
    try:
        with open(launcher_path, "w", encoding="utf-8") as f:
            f.write(launcher_content)
        print("Launcher script created successfully.")
    except Exception as e:
        print(f"[Error] Failed to write launcher script: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3.5. Copy static Web UI assets to dist/zipvoice_onnx/web so PyInstaller can find and bundle them
    src_web = os.path.abspath("src/zipvoice_onnx/web")
    dest_web = os.path.join(dist_dir, "zipvoice_onnx", "web")
    if os.path.exists(src_web):
        print(f"\nCopying static Web UI assets to {dest_web}...")
        try:
            shutil.copytree(src_web, dest_web)
            print("Web assets copied successfully.")
        except Exception as e:
            print(f"[Error] Failed to copy web assets: {e}", file=sys.stderr)
            sys.exit(1)
            
    # 4. Run PyInstaller to build the executable
    print("\nRunning PyInstaller to compile executable...")
    icon_path = os.path.abspath("icon.ico")
    cmd_pyinstaller = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onedir",
        "--console",
        "--icon", icon_path if os.path.exists(icon_path) else None,
        "--paths", "dist",
        "--collect-data", "language_tags",
        "--collect-data", "g2p_en",
        "--collect-data", "pypinyin",
        "--collect-data", "whisper",
        "--collect-all", "espeakng_loader",
        "--add-data", "dist/zipvoice_onnx/web;zipvoice_onnx/web",
        "--add-data", "dist/zipvoice_onnx/pyarmor_runtime_000000;zipvoice_onnx/pyarmor_runtime_000000",
        launcher_path
    ]
    # Filter out None values
    cmd_pyinstaller = [arg for arg in cmd_pyinstaller if arg is not None]
    
    print(f"Executing command: {' '.join(cmd_pyinstaller)}")
    try:
        subprocess.run(cmd_pyinstaller, check=True)
        print("PyInstaller compilation completed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n[Error] PyInstaller packaging failed!", file=sys.stderr)
        sys.exit(1)
        
    # 5. Copy model directories and audio examples to the output folder
    exe_dist_dir = os.path.join(dist_dir, "run_server")
    
    # Model directory (model-en-distilled)
    src_model = os.path.abspath("model-en-distilled")
    dest_model = os.path.join(exe_dist_dir, "model-en-distilled")
    if os.path.exists(src_model):
        print(f"\nCopying model folder from {src_model} to {dest_model}...")
        try:
            shutil.copytree(src_model, dest_model)
            print("Model files copied successfully.")
        except Exception as e:
            print(f"[Warning] Failed to copy model files: {e}. You may need to copy them manually.")
    else:
        print(f"\n[Warning] Model folder not found at {src_model}!")

    # Model directory (generic model if exists)
    src_model_generic = os.path.abspath("model")
    dest_model_generic = os.path.join(exe_dist_dir, "model")
    if os.path.exists(src_model_generic):
        print(f"\nCopying generic model folder from {src_model_generic} to {dest_model_generic}...")
        try:
            shutil.copytree(src_model_generic, dest_model_generic)
            print("Generic model files copied successfully.")
        except Exception as e:
            print(f"[Warning] Failed to copy generic model files: {e}.")

    # Audio directory
    src_audio = os.path.abspath("examples/audio")
    dest_audio = os.path.join(exe_dist_dir, "examples/audio")
    if os.path.exists(src_audio):
        print(f"Copying reference audio from {src_audio} to {dest_audio}...")
        try:
            shutil.copytree(src_audio, dest_audio)
            print("Reference audio copied successfully.")
        except Exception as e:
            print(f"[Warning] Failed to copy reference audio: {e}.")
            
    print("\n=== Package Build Completed Successfully! ===")
    print(f"Standalone executable directory: {exe_dist_dir}")
    print(f"Executable file: {os.path.join(exe_dist_dir, 'run_server.exe')}")
    print("\nYou can run the server directly by launching the exe:")
    print(f"  {os.path.join(exe_dist_dir, 'run_server.exe')}")

if __name__ == "__main__":
    main()
