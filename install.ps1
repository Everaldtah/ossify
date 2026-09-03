# Ossify installer: npm deps, PowerShell profile hook, and cmd shims on PATH.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\install.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "[ossify] repo: $root"
foreach ($tool in @("node", "npm", "claude")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not found on PATH. Need Node 20+ and Claude Code." }
}
$lms = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"
if (-not (Test-Path $lms)) { Write-Warning "LM Studio CLI (lms) not found at $lms. Install LM Studio, open it once, and download the models you want." }

Push-Location $root
try { npm install --omit=dev --silent } finally { Pop-Location }

# 1) PowerShell profile: dot-source the launchers (idempotent).
$profilePath = $PROFILE
$profileDir = Split-Path -Parent $profilePath
if (-not (Test-Path $profileDir)) { New-Item -ItemType Directory -Force $profileDir | Out-Null }
if (-not (Test-Path $profilePath)) { New-Item -ItemType File $profilePath | Out-Null }
$marker = "# ossify - Claude Code on local LM Studio models (gptoss / qwen35)"
$content = Get-Content $profilePath -Raw -ErrorAction SilentlyContinue
if (-not $content -or -not $content.Contains($marker)) {
    Copy-Item $profilePath "$profilePath.bak-ossify" -Force
    $hook = "`r`n$marker`r`n. `"$root\bin\ossify.ps1`"`r`n"
    [IO.File]::AppendAllText($profilePath, $hook, [Text.Encoding]::ASCII)
    Write-Host "[ossify] added launcher hook to $profilePath (backup: .bak-ossify)"
} else { Write-Host "[ossify] profile hook already present" }

# 2) cmd shims in ~\.local\bin (same folder Claude Code uses), added to the user PATH if needed.
$binDir = Join-Path $env:USERPROFILE ".local\bin"
if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Force $binDir | Out-Null }
foreach ($shim in @("gptoss.cmd", "qwen35.cmd")) {
    $target = Join-Path $binDir $shim
    "@echo off`r`npowershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$root\bin\run.ps1`" $($shim.Replace('.cmd','')) %*`r`n" | Set-Content -Path $target -Encoding Ascii
}
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not (($userPath -split ";") -contains $binDir)) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$binDir", "User")
    Write-Host "[ossify] added $binDir to your user PATH (open a new terminal to pick it up)"
}

Write-Host ""
Write-Host "Installed. Open a new terminal and type:" -ForegroundColor Green
Write-Host "   gptoss          Claude Code on gpt-oss-20b"
Write-Host "   qwen35          Claude Code on Qwen3.5-35B-A3B"
Write-Host "   gptoss --oss-help    all options"
Write-Host "First run of each model auto-tunes GPU/CPU placement for this PC (a few minutes, once)."
