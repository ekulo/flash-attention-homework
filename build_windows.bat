@echo off
REM =====================================================================
REM  FlashAttention Homework - one-shot build + test + bench (Windows)
REM  Requirement: VS2022 x64 + CUDA Toolkit >= 12.8 + Python(torch)
REM  Usage: build_windows.bat [arch]   (default arch=120 for RTX 50)
REM  If you see 'cmake' or 'cl' not found, run this from the
REM  "x64 Native Tools Command Prompt for VS 2022".
REM =====================================================================
setlocal
set ARCH=%~1
if "%ARCH%"=="" set ARCH=120
echo == FlashAttention: arch=%ARCH% ==
python scripts\run_all.py --arch %ARCH% %2 %3 %4 %5 %6
endlocal
