@echo off
rem qwfnfer.cmd: the one command on Windows, the counterpart of scripts/qwfnfer.
rem Starts the console (or finds the one already running) and opens it in the
rem browser. Works from a release bundle (this file next to bin\ and tools\) and
rem from a source checkout (this file under scripts\).
rem
rem   qwfnfer                 open the console at http://127.0.0.1:8090
rem   qwfnfer --no-browser    just run the console in this terminal
rem   All other arguments pass through to the console (--port N, --server-port N,
rem   --start, --model NAME, --preset NAME).
setlocal
set "HERE=%~dp0"
if exist "%HERE%tools\qwfn_console.py" goto :found
for %%I in ("%HERE%..") do set "HERE=%%~fI\"
if exist "%HERE%tools\qwfn_console.py" goto :found
echo qwfnfer: tools\qwfn_console.py not found next to %~f0 1>&2
exit /b 1
:found
cd /d "%HERE%"

set "port=8090"
set "browser=1"
set "args="
:parse
if "%~1"=="" goto :run
if /i "%~1"=="--no-browser" (
    set "browser=0"
    shift
    goto :parse
)
if /i "%~1"=="--port" (
    set "port=%~2"
    set "args=%args% %~1 %~2"
    shift
    shift
    goto :parse
)
set "args=%args% %~1"
shift
goto :parse

:run
set "url=http://127.0.0.1:%port%"
rem Already running? The console port answers; just open it.
powershell -NoProfile -Command "$c = New-Object Net.Sockets.TcpClient; try { $c.Connect('127.0.0.1', %port%); exit 0 } catch { exit 1 } finally { $c.Close() }" >nul 2>&1
if not errorlevel 1 (
    echo qwfnfer console is already running at %url%
    if "%browser%"=="1" start "" "%url%"
    exit /b 0
)
rem Open the browser a beat after the console is up, then run the console.
if "%browser%"=="1" start "" /b cmd /c "timeout /t 2 /nobreak >nul & start %url%"
python tools\qwfn_console.py %args%
