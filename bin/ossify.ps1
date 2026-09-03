# Ossify launchers for Claude Code on local LM Studio models.
#   gptoss   -> openai/gpt-oss-20b        (64k context, needs ~10 GB free RAM)
#   qwen35   -> qwen/qwen3.5-35b-a3b      (64k context, needs ~21 GB free RAM)
# Plain `claude` is untouched: provider env vars are set only for the child process and restored after.
# Pure ASCII, no BOM on purpose (some AV engines lock BOM'd .ps1 files).

$script:OssifyRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $script:OssifyRoot "src\cli.mjs"))) { $script:OssifyRoot = Join-Path $env:USERPROFILE "ossify" }

function Show-OssifyHelp($name, $model) {
    Write-Host ""
    Write-Host "$name - Claude Code on $model via LM Studio (Ossify)" -ForegroundColor Cyan
    Write-Host "  $name                     load the model (tuned for this PC) and start Claude Code"
    Write-Host "  $name -p 'prompt'         any normal claude args pass straight through"
    Write-Host "  $name --oss-ctx 32768     use a different context length"
    Write-Host "  $name --oss-status        server / GPU / RAM / loaded model / tuned profile"
    Write-Host "  $name --oss-tune          re-run the speed auto-tuner   (--oss-deep: more candidates)"
    Write-Host "  $name --oss-plan          show what the planner would do without loading"
    Write-Host "  $name --oss-bench         benchmark the currently loaded model"
    Write-Host "  $name --oss-unload        unload everything, free VRAM and RAM"
    Write-Host "  $name --oss-reset         reload with an empty prompt cache (if replies quote an older chat)"
    Write-Host "  $name --oss-quick         skip auto-tune on first run (planner default)"
    Write-Host "  $name --oss-retune        ignore the saved profile and tune again"
    Write-Host "  $name --oss-ttl 0         keep the model loaded forever (default: unload after 30 min idle)"
    Write-Host "  $name --oss-doctor        environment dump for bug reports"
    Write-Host ""
}

function Start-OssifyClaude($ProfileName, $Model, $DefaultCtx, $RamMarginGB, $ArgList) {
    $cli = Join-Path $script:OssifyRoot "src\cli.mjs"
    if (-not (Test-Path $cli)) { Write-Host "[ossify] cli not found at $cli - run install.ps1 from the ossify repo" -ForegroundColor Red; return }
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Write-Host "[ossify] node.exe not on PATH (need Node 20+)" -ForegroundColor Red; return }

    $ctx = $DefaultCtx; $ttl = 1800; $cmd = $null; $extra = @(); $passthru = @()
    $list = @($ArgList)
    for ($i = 0; $i -lt $list.Count; $i++) {
        switch ($list[$i]) {
            '--oss-ctx'    { $i++; $ctx = [int]$list[$i] }
            '--oss-ttl'    { $i++; $ttl = [int]$list[$i] }
            '--oss-tune'   { $cmd = 'tune' }
            '--oss-deep'   { $cmd = 'tune'; $extra += '--deep' }
            '--oss-status' { $cmd = 'status' }
            '--oss-unload' { $cmd = 'unload' }
            '--oss-reset'  { $cmd = 'reset' }
            '--oss-plan'   { $cmd = 'plan' }
            '--oss-bench'  { $cmd = 'bench' }
            '--oss-doctor' { $cmd = 'doctor' }
            '--oss-quick'  { $extra += '--quick' }
            '--oss-retune' { $extra += '--retune' }
            '--oss-help'   { Show-OssifyHelp $ProfileName $Model; return }
            default        { $passthru += $list[$i] }
        }
    }
    $common = @('--model', $Model, '--ctx', "$ctx", '--ttl', "$ttl", '--ram-margin', "$RamMarginGB") + $extra
    if ($cmd) { & node $cli $cmd @common; return }

    & node $cli up @common
    if ($LASTEXITCODE -ne 0) { Write-Host "[ossify] model is not loaded - not starting Claude Code." -ForegroundColor Red; return }

    $cur = Get-Content (Join-Path $env:USERPROFILE ".ossify\current.json") -Raw | ConvertFrom-Json
    $compact = [Math]::Max(16384, [int]$cur.contextLength - 8192)
    $vars = [ordered]@{
        ANTHROPIC_BASE_URL                      = $cur.baseUrl
        ANTHROPIC_AUTH_TOKEN                    = "lm-studio"
        ANTHROPIC_MODEL                         = $cur.identifier
        ANTHROPIC_DEFAULT_OPUS_MODEL            = $cur.identifier
        ANTHROPIC_DEFAULT_SONNET_MODEL          = $cur.identifier
        ANTHROPIC_DEFAULT_HAIKU_MODEL           = $cur.identifier
        ANTHROPIC_SMALL_FAST_MODEL              = $cur.identifier
        CLAUDE_CODE_SUBAGENT_MODEL              = $cur.identifier
        CLAUDE_CODE_AUTO_COMPACT_WINDOW         = "$compact"
        CLAUDE_CODE_MAX_OUTPUT_TOKENS           = "8192"
        API_TIMEOUT_MS                          = "3600000"
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
        DISABLE_TELEMETRY                       = "1"
        DISABLE_ERROR_REPORTING                 = "1"
        # Clear anything that would hijack the request to another provider, for this run only:
        ANTHROPIC_API_KEY                       = $null
        CLAUDE_CODE_USE_OPENAI                  = $null
        OPENAI_BASE_URL                         = $null
        OPENAI_API_KEY                          = $null
        CLAUDE_CODE_USE_BEDROCK                 = $null
        CLAUDE_CODE_USE_VERTEX                  = $null
    }
    $saved = @{}
    foreach ($k in $vars.Keys) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        if ($null -eq $vars[$k]) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } else { Set-Item "Env:$k" $vars[$k] }
    }
    Write-Host "[ossify] Claude Code -> $($cur.identifier) @ $($cur.baseUrl)  (ctx $($cur.contextLength), $($cur.strategy))" -ForegroundColor DarkGray
    try {
        $claude = Get-Command claude -ErrorAction SilentlyContinue
        if ($claude) { & $claude.Source @passthru } else { & "$env:USERPROFILE\.local\bin\claude.exe" @passthru }
    }
    finally {
        foreach ($k in $vars.Keys) {
            if ($null -eq $saved[$k]) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } else { Set-Item "Env:$k" $saved[$k] }
        }
    }
}

function gptoss { Start-OssifyClaude 'gptoss' 'openai/gpt-oss-20b' 65536 4 $args }
function qwen35 { Start-OssifyClaude 'qwen35' 'qwen/qwen3.5-35b-a3b' 65536 2 $args }
