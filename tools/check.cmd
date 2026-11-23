@echo off
rem Local verification on native Windows (build + tests, both under tools\guard.py).
rem   tools\check.cmd
setlocal
cd /d "%~dp0.."
call tools\build.cmd || ( echo build failed & exit /b 1 )
if not defined PY set PY=python
set TORCHINDUCTOR_COMPILE_THREADS=2
set OMP_NUM_THREADS=2
%PY% tools\guard.py --log logs\tests_win.csv -- %PY% -m pytest -q tests\ -p no:cacheprovider -rs
if errorlevel 1 ( echo CHECK FAILED & exit /b 1 )
echo CHECK PASSED
