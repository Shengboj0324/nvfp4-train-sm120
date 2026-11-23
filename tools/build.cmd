@echo off
rem Build the CUDA extension in place on native Windows. Usage: tools\build.cmd
rem Needs Visual Studio 2022 (C++ workload), CUDA >= 12.8 and a CUDA build of PyTorch >= 2.7 in the
rem active Python. The compile runs under tools\guard.py, like the Linux build.
setlocal
cd /d "%~dp0.."
if not defined CUDA_HOME if defined CUDA_PATH set "CUDA_HOME=%CUDA_PATH%"
if not defined CUDA_HOME (
  for /d %%D in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.*") do set "CUDA_HOME=%%~D"
)
if not defined CUDA_HOME ( echo CUDA toolkit not found; set CUDA_HOME & exit /b 1 )
set "PATH=%CUDA_HOME%\bin;%PATH%"
for /f "usebackq delims=" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%I"
if not defined VS ( echo Visual Studio with the C++ workload not found & exit /b 1 )
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set DISTUTILS_USE_SDK=1
if not defined MAX_JOBS set MAX_JOBS=4
if not defined PY set PY=python
%PY% tools\guard.py --log logs\build_win.csv --max-load-per-cpu 3 -- %PY% setup.py build_ext --inplace
