# Build the relocatable Windows release bundle: dist\qwfnfer-windows-x86_64-cuda.zip
#
# The Windows counterpart of scripts/package.sh. The bundle carries everything but the NVIDIA
# driver and Python: the engine and the tokenizer tool (QWFN_PORTABLE: AVX2 code, the libraries
# next to the binaries), ggml/llama.cpp's DLLs from a portable build (GGML_NATIVE=OFF, every CPU
# variant, CUDA architectures 75-120), the CUDA runtime DLLs NVIDIA lets applications redistribute
# (cudart, cublas, cublasLt), the Visual C++ runtime (app-local, as Microsoft allows, so the user
# needs no Redistributable), the console and a launcher.
#
# Run it from a "x64 Native Tools Command Prompt for VS 2022" (or after vcvars64.bat), with CMake,
# Ninja and the CUDA toolkit installed:
#
#   powershell -ExecutionPolicy Bypass -File scripts\package.ps1
#   set GGML_LIBS=D:\ggml\bin & set VERSION=v0.3 & powershell -ExecutionPolicy Bypass -File scripts\package.ps1
#
# Inputs (environment variables):
#   LLAMA_CPP_ROOT    llama.cpp source (ggml headers, vendor\)   default %USERPROFILE%\.unsloth\llama.cpp
#   GGML_LIBS         the portable ggml/llama DLLs                 default %USERPROFILE%\.cache\qwfnfer-build\ggml\bin
#                     Built once from Unsloth's llama.cpp at commit ca14269 (README, "Building from
#                     source"), library targets only, in the same developer prompt:
#
#                       cmake -S %USERPROFILE%\.unsloth\llama.cpp -B %USERPROFILE%\.cache\qwfnfer-build\ggml -G Ninja ^
#                         -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON ^
#                         -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON ^
#                         -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120" ^
#                         -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^
#                         -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
#                       cmake --build %USERPROFILE%\.cache\qwfnfer-build\ggml
#
#   CUDA_PATH         the CUDA toolkit (its installer sets it); the runtime DLLs come from bin\x64
#                     (CUDA 13) or bin
#   VCToolsRedistDir  set by the developer prompt: where the Visual C++ runtime DLLs are taken from
#   VERSION           default: git describe
#
# Before it zips anything it checks that every DLL the bundled binaries import is either in the
# bundle, part of Windows, or the NVIDIA driver's nvcuda.dll: dumpbin /dependents on each one.

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Env-Or([string] $name, [string] $default) {
    $v = [Environment]::GetEnvironmentVariable($name)
    if ($v) { return $v } else { return $default }
}

$LlamaRoot = Env-Or 'LLAMA_CPP_ROOT' (Join-Path $env:USERPROFILE '.unsloth\llama.cpp')
$GgmlLibs  = Env-Or 'GGML_LIBS'      (Join-Path $env:USERPROFILE '.cache\qwfnfer-build\ggml\bin')
$Version   = Env-Or 'VERSION' ''
# Native commands whose stderr is redirected go through cmd: Windows PowerShell 5.1 turns
# redirected stderr into errors, and with ErrorActionPreference Stop those end the script.
if (-not $Version -and (Get-Command git -ErrorAction SilentlyContinue)) { $Version = (cmd /c 'git describe --tags --always --dirty 2>nul') }
if (-not $Version) { $Version = Get-Date -Format 'yyyyMMdd' }
$Name  = 'qwfnfer-windows-x86_64-cuda'
$Out   = Join-Path 'dist' $Name
$Build = 'build-portable-win'

foreach ($tool in 'cmake', 'ninja', 'dumpbin') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is not on PATH: run this from a x64 Native Tools Command Prompt for VS 2022, with CMake and Ninja installed"
    }
}
if (-not (Test-Path (Join-Path $GgmlLibs 'ggml-cuda.dll'))) {
    throw "no portable ggml build at $GgmlLibs (ggml-cuda.dll missing): see the header of this script"
}
if (-not $env:CUDA_PATH) { throw 'CUDA_PATH is not set: install the CUDA toolkit, or point CUDA_PATH at it' }

Write-Host "== engine (portable build against $GgmlLibs)"
cmd /c "cmake -S . -B $Build -G Ninja -DCMAKE_BUILD_TYPE=Release -DQWFN_PORTABLE=ON `"-DLLAMA_CPP_ROOT=$LlamaRoot`" `"-DLLAMA_CPP_BUILD=$GgmlLibs`" > $Build.cmake.log 2>&1"
if ($LASTEXITCODE) { Get-Content "$Build.cmake.log" -Tail 20; throw 'cmake configure failed' }
cmake --build $Build --target qwfn-server qwfn-tok qwfn-io-test
if ($LASTEXITCODE) { throw 'build failed' }

# The io layer is the part of the engine that is different on Windows: a build whose reads are
# wrong does not get packaged.
Write-Host '== io layer tests'
& (Join-Path $Build 'qwfn-io-test.exe') | Select-Object -Last 1
if ($LASTEXITCODE) { throw "qwfn-io-test failed: run $Build\qwfn-io-test.exe for the details" }

Write-Host "== bundle $Out"
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force "$Out\bin", "$Out\tools\console" | Out-Null
Copy-Item (Join-Path $Build 'qwfn-server.exe'), (Join-Path $Build 'qwfn-tok.exe') "$Out\bin\"
# ggml, its backends (the CPU one is several DLLs, one per ISA level) and llama
Copy-Item (Join-Path $GgmlLibs 'ggml*.dll'), (Join-Path $GgmlLibs 'llama.dll') "$Out\bin\"

$cudaBin = @('bin\x64', 'bin') | ForEach-Object { Join-Path $env:CUDA_PATH $_ } |
           Where-Object { Test-Path (Join-Path $_ 'cublas64_*.dll') } | Select-Object -First 1
if (-not $cudaBin) { throw "no cublas64_*.dll under $env:CUDA_PATH\bin\x64 or $env:CUDA_PATH\bin" }
foreach ($pat in 'cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll') {
    Copy-Item (Join-Path $cudaBin $pat) "$Out\bin\"
}

# The Visual C++ runtime, app-local: msvcp140/vcruntime140 and, for ggml-cpu's OpenMP, vcomp140.
if ($env:VCToolsRedistDir) {
    foreach ($kind in 'CRT', 'OpenMP') {
        $d = Get-ChildItem (Join-Path $env:VCToolsRedistDir 'x64') -Directory -Filter "Microsoft.VC*.$kind" -ErrorAction SilentlyContinue |
             Select-Object -First 1
        if ($d) { Copy-Item (Join-Path $d.FullName '*.dll') "$Out\bin\" }
    }
}

# Every import of every bundled binary must resolve on a machine that has Windows and the NVIDIA
# driver and nothing else. The Visual C++ runtime is on the build machine's System32 but not
# necessarily on the user's, so it counts only when it is in the bundle.
$have = @(Get-ChildItem "$Out\bin" -File | ForEach-Object { $_.Name.ToLower() })
$missing = @()
foreach ($f in Get-ChildItem "$Out\bin" -File | Where-Object { $_.Extension -in '.exe', '.dll' }) {
    foreach ($line in (dumpbin /nologo /dependents $f.FullName)) {
        if ($line -notmatch '^\s+(\S+\.dll)\s*$') { continue }
        $dep = $Matches[1].ToLower()
        if ($have -contains $dep) { continue }
        if ($dep -eq 'nvcuda.dll' -or $dep -like 'api-ms-win-*' -or $dep -like 'ext-ms-*') { continue }
        $runtime = $dep -like 'vcruntime*' -or $dep -like 'msvcp*' -or $dep -like 'vcomp*' -or $dep -like 'concrt*'
        if (-not $runtime -and (Test-Path (Join-Path $env:SystemRoot "System32\$dep"))) { continue }
        $missing += "$($f.Name) -> $dep"
    }
}
if ($missing) {
    throw ("the bundle would not start on a clean machine; missing:`n  " + ($missing -join "`n  ") +
           "`n(the Visual C++ runtime comes from VCToolsRedistDir: run from a VS developer prompt)")
}

Copy-Item tools\qwfn_console.py, tools\qwfn_router.py "$Out\tools\"
Copy-Item tools\console\index.html "$Out\tools\console\"
Copy-Item README.md, LICENSE "$Out\"
Set-Content -Path "$Out\VERSION" -Value $Version -Encoding ascii

# The launcher. The console opens on port 8090; the browser is pointed at it once it is up.
$launcher = @'
@echo off
rem qwfnfer: the console at http://127.0.0.1:8090. Ctrl+C stops it and the server it started.
rem Arguments go to the console: --start starts the last served model and tier right away.
setlocal
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo qwfnfer needs Python 3: https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
  exit /b 1
)
start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start "" http://127.0.0.1:8090"
%PY% "%~dp0tools\qwfn_console.py" %*
'@
Set-Content -Path "$Out\qwfnfer.cmd" -Value $launcher -Encoding ascii

$install = @"
qwfnfer $Version -- Qwen3.8-Flash-Next on one 16 GB GPU (Windows 10/11 x64, NVIDIA)

1. Unzip anywhere and run:   qwfnfer.cmd
   (opens the console at http://127.0.0.1:8090; needs Python 3 and an NVIDIA driver 580 or newer)
2. Get the model once:       hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "mmproj-F16.gguf"
   (pip install -U huggingface_hub for the hf command; the console finds the files in the Hugging Face cache)
3. Pick a tier (Chat, Agentic coding, Agentic coding+ or your own Custom one) and press Auto-tune & start:
   the console measures your drive, threads and memory, picks every flag, starts the server and verifies it.
   The endpoint is http://127.0.0.1:8080/v1. Model locations: add any folder that holds the shards.
For full speed, in the NVIDIA Control Panel (Manage 3D settings, Program settings, add bin\qwfn-server.exe)
set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback", and turn on Hardware-accelerated GPU
scheduling (Settings, System, Display, Graphics). README.md has the reasons.
Everything the engine needs is in bin\ except the NVIDIA driver. See README.md.
"@
Set-Content -Path "$Out\INSTALL.txt" -Value $install -Encoding ascii

$zip = Join-Path 'dist' "$Name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
# tar (bsdtar, in Windows since 10 1803) writes zips past Compress-Archive's 2 GB limits.
tar -a -c -f $zip -C dist $Name
if ($LASTEXITCODE) { throw 'zip failed' }
Write-Host ("wrote {0} ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
