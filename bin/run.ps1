# Entry point for the .cmd shims (cmd.exe / other terminals): run.ps1 <gptoss|qwen35> [claude args]
param([Parameter(Mandatory = $true)][string]$Launcher, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
. (Join-Path $PSScriptRoot "ossify.ps1")
$global:LASTEXITCODE = 0
& $Launcher @Rest
if ($null -eq $LASTEXITCODE) { exit 0 }
exit $LASTEXITCODE
