@echo off
echo ==================================================
echo         ZipVoice ONNX Build and Package Tools
echo ==================================================
echo.

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Found uv tool. Running build.py using uv...
    uv run python build.py
) else (
    echo [INFO] uv not found. Running build.py using system python...
    python build.py
)

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Package completed successfully!
    echo [INFO] The packaged files are in: dist\run_server
    echo [INFO] Run dist\run_server\run_server.exe to start the server.
) else (
    echo.
    echo [ERROR] Package build failed. Please check the logs.
)

echo.
pause
