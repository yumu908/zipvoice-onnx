@echo off
:: -------------------------------------------------
:: �� ����ԴĿ¼����� cuDNN ��װĿ¼����Ŀ�� CUDA Ŀ¼
:: -------------------------------------------------
set CUDNN_ROOT=C:\Program Files\NVIDIA\CUDNN\v9.24
set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

:: -------------------------------------------------
:: �� ���� bin��include��lib\x64 ������Ŀ¼
:: -------------------------------------------------
echo ���ڸ��� cuDNN bin��include��lib �� CUDA Ŀ¼ ...
xcopy /e /i /y "%CUDNN_ROOT%\bin"   "%CUDA_ROOT%\bin"   > nul
xcopy /e /i /y "%CUDNN_ROOT%\include" "%CUDA_ROOT%\include" > nul
xcopy /e /i /y "%CUDNN_ROOT%\lib\x64" "%CUDA_ROOT%\lib\x64" > nul
echo ������ɡ�
pause
