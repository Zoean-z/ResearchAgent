[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$BindPort = 8000,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

function Test-ProcessAlive {
    param([int]$ProcessId)
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        return -not $process.HasExited
    } catch {
        return $false
    }
}

function Stop-ManagedProcess {
    param([int]$ProcessId)

    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    } catch {
        Write-Warning ("Failed to stop existing backend process {0}: {1}" -f $ProcessId, $_.Exception.Message)
    }
}

function Ensure-RepoLocalOpenVikingConfig {
    param(
        [string]$RepoRoot,
        [string]$ConfigPath
    )

    $configDir = Split-Path -Parent $ConfigPath
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null

    $sourceConfig = Join-Path $HOME ".openviking\ov.conf"
    $exampleConfig = Join-Path $RepoRoot "data\openviking\ov.conf.example"

    if (Test-Path $ConfigPath) {
        $raw = Get-Content $ConfigPath -Raw
        $config = $raw | ConvertFrom-Json
    } elseif (Test-Path $sourceConfig) {
        $raw = Get-Content $sourceConfig -Raw
        $config = $raw | ConvertFrom-Json
    } elseif (Test-Path $exampleConfig) {
        Copy-Item $exampleConfig $ConfigPath -Force
        $config = (Get-Content $ConfigPath -Raw) | ConvertFrom-Json
    } else {
        throw "Missing OpenViking config. Copy `$HOME\.openviking\ov.conf` to `data\openviking\ov.conf` first."
    }

    if (-not $config.storage) {
        $config | Add-Member -NotePropertyName storage -NotePropertyValue ([pscustomobject]@{}) -Force
    }
    $config.storage.workspace = (Join-Path $RepoRoot "data\openviking_workspace")

    $config | ConvertTo-Json -Depth 20 | Set-Content -Path $ConfigPath -Encoding utf8
}

function Ensure-OpenVikingInstalled {
    $probe = & py -3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('openviking') else 1)"
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Host "Installing openviking into the active Python environment..."
    & py -3 -m pip install openviking --upgrade --force-reinstall
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install the openviking package."
    }

    & py -3 -c "import openviking"
    if ($LASTEXITCODE -ne 0) {
        throw "openviking still cannot be imported after installation."
    }
}

function Import-DotEnvToProcess {
    param([string]$Path)

    foreach ($rawLine in Get-Content $Path -Encoding utf8) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        $separator = $line.IndexOf("=")
        if ($separator -lt 1) {
            continue
        }
        $key = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendRoot = Join-Path $repoRoot "backend"
$envPath = Join-Path $repoRoot ".env"
$openvikingConfigPath = Join-Path $repoRoot "data\openviking\ov.conf"
$logDir = Join-Path $repoRoot ".project-loop\run"
$pidFile = Join-Path $logDir "backend.pid"
$stdoutLog = Join-Path $logDir "backend.out.log"
$stderrLog = Join-Path $logDir "backend.err.log"

if (-not (Test-Path $envPath)) {
    throw "Missing `.env`. Copy `.env.example` to `.env` first."
}

Import-DotEnvToProcess -Path $envPath
Ensure-RepoLocalOpenVikingConfig -RepoRoot $repoRoot -ConfigPath $openvikingConfigPath

$env:RESEARCH_AGENT_ENV_FILE = $envPath
$env:RESEARCH_AGENT_OPENVIKING_BACKEND = "embedded"
$env:RESEARCH_AGENT_OPENVIKING_DATA_PATH = (Join-Path $repoRoot "data\openviking")
$env:OPENVIKING_CONFIG_FILE = $openvikingConfigPath

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (Test-Path $pidFile) {
    $existingPid = [int](Get-Content $pidFile -Raw).Trim()
    if ($existingPid -gt 0 -and (Test-ProcessAlive -ProcessId $existingPid)) {
        if ($Restart) {
            Write-Host "Restarting existing backend PID $existingPid ..."
            Stop-ManagedProcess -ProcessId $existingPid
            Start-Sleep -Seconds 1
        } else {
        Write-Host "Backend already running as PID $existingPid."
        Write-Host "Health: http://$BindHost`:$BindPort/health"
        exit 0
        }
    }
}

try {
    Ensure-OpenVikingInstalled
} catch {
    throw $_
}

$arguments = @(
    "-3",
    "-m",
    "uvicorn",
    "research_agent.api.app:app",
    "--app-dir",
    "backend",
    "--host",
    $BindHost,
    "--port",
    "$BindPort"
)

$process = Start-Process `
    -FilePath "py" `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog

Set-Content -Path $pidFile -Value $process.Id -Encoding ascii

$healthUrl = "http://$BindHost`:$BindPort/health"
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    if ($process.HasExited) {
        Write-Host "Backend exited early. Tail of stdout:"
        if (Test-Path $stdoutLog) {
            Get-Content $stdoutLog -Tail 40
        }
        Write-Host "Tail of stderr:"
        if (Test-Path $stderrLog) {
            Get-Content $stderrLog -Tail 40
        }
        throw "Backend failed to start."
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($health.status -eq "ok") {
            Write-Host "Backend started successfully."
            Write-Host "PID: $($process.Id)"
            Write-Host "Health: $healthUrl"
            Write-Host "Logs: $stdoutLog, $stderrLog"
            exit 0
        }
    } catch {
        continue
    }
}

throw "Timed out waiting for backend health check at $healthUrl."
