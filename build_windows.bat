@echo off
echo Building ultra-fast NCD worker for Windows...
echo.
echo NOTE: This requires MinGW-w64 or MSYS2 with g++ installed
echo If you don't have it, use WSL instead (see RUN_INSTRUCTIONS.md)
echo.

g++ -O3 -march=native -flto -fopenmp -pthread ncd_worker.cpp -llzma -lxxhash -o ncd_worker.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful! → ncd_worker.exe
) else (
    echo Build failed!
    echo.
    echo Install dependencies using MSYS2:
    echo   1. Download MSYS2 from https://www.msys2.org/
    echo   2. Open MSYS2 terminal and run:
    echo      pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-xz mingw-w64-x86_64-xxhash
    echo.
    echo OR use WSL (recommended):
    echo   wsl --install
    echo   Then follow Linux instructions in RUN_INSTRUCTIONS.md
)
