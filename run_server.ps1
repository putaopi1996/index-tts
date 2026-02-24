# 设置变量
$env:UV_SKIP_WHEEL_FILENAME_CHECK="1"

$ErrorActionPreference = "Stop"

# Get script directory
$scriptPath = $PSScriptRoot
if (-not $scriptPath) {
    $scriptPath = Get-Location
}
Set-Location $scriptPath

Write-Host "Starting IndexTTS2 API Server..." -ForegroundColor Green

# Try uv
if (Get-Command "uv" -ErrorAction SilentlyContinue) {
    Write-Host "Detected 'uv', running with 'uv run --no-sync'..." -ForegroundColor Cyan
    # --- 修改这里：加上 --no-sync ---
    uv run --no-sync python api_server.py
    # -----------------------------
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Server exited with error code $LASTEXITCODE" -ForegroundColor Red
        Read-Host "Press Enter to exit..."
    }
    exit
}

# Try python directly
Write-Host "'uv' not found, trying 'python'..." -ForegroundColor Cyan
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    python api_server.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Server exited with error code $LASTEXITCODE" -ForegroundColor Red
        Read-Host "Press Enter to exit..."
    }
    exit
}

Write-Host "Error: Neither 'uv' nor 'python' command found." -ForegroundColor Red
Read-Host "Press Enter to exit..."