[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$artifacts = @(
    [pscustomobject]@{
        Name = 'YOLO11m detector'
        Url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt'
        RelativePath = 'models/yolo/yolo11m.pt'
        Source = 'Ultralytics assets release v8.3.0'
    },
    [pscustomobject]@{
        Name = 'FastReID MSMT17 SBS R50-IBN weights'
        Url = 'https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50-ibn.pth'
        RelativePath = 'models/fastreid/msmt17_sbs_R50-IBN.pth'
        Source = 'FastReID official model zoo release v0.1.1'
    },
    [pscustomobject]@{
        Name = 'FastReID VeRi SBS R50-IBN weights'
        Url = 'https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth'
        RelativePath = 'models/fastreid/veri_sbs_R50-IBN.pth'
        Source = 'FastReID official model zoo release v0.1.1'
    },
    [pscustomobject]@{
        Name = 'FastReID MSMT17 SBS R50-IBN config'
        Url = 'https://raw.githubusercontent.com/JDAI-CV/fast-reid/v0.1.1/configs/MSMT17/sbs_R50-ibn.yml'
        RelativePath = 'models/fastreid/configs/msmt17_sbs_R50-IBN.yml'
        Source = 'FastReID official repository tag v0.1.1'
    },
    [pscustomobject]@{
        Name = 'FastReID VeRi SBS R50-IBN config'
        Url = 'https://raw.githubusercontent.com/JDAI-CV/fast-reid/v0.1.1/configs/VeRi/sbs_R50-ibn.yml'
        RelativePath = 'models/fastreid/configs/veri_sbs_R50-IBN.yml'
        Source = 'FastReID official repository tag v0.1.1'
    },
    [pscustomobject]@{
        Name = 'FastReID Base-Strongerbaseline config'
        Url = 'https://raw.githubusercontent.com/JDAI-CV/fast-reid/v0.1.1/configs/Base-Strongerbaseline.yml'
        RelativePath = 'models/fastreid/Base-Strongerbaseline.yml'
        Source = 'FastReID official repository tag v0.1.1'
    },
    [pscustomobject]@{
        Name = 'FastReID Base-bagtricks config'
        Url = 'https://raw.githubusercontent.com/JDAI-CV/fast-reid/v0.1.1/configs/Base-bagtricks.yml'
        RelativePath = 'models/fastreid/Base-bagtricks.yml'
        Source = 'FastReID official repository tag v0.1.1'
    }
)

function Get-AbsolutePath([string]$relativePath) {
    return Join-Path $repoRoot ($relativePath -replace '/', [IO.Path]::DirectorySeparatorChar)
}

function Get-Sha256([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Download-Artifact($artifact) {
    $destination = Get-AbsolutePath $artifact.RelativePath
    $parent = Split-Path -Parent $destination
    $partial = "$destination.download"
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    if ($Force -and (Test-Path -LiteralPath $partial -PathType Leaf)) {
        Remove-Item -LiteralPath $partial -Force
    }

    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        if (-not $Force) {
            $hash = Get-Sha256 $destination
            Write-Host "SKIP  $($artifact.Name)"
            Write-Host "      $($artifact.RelativePath)"
            Write-Host "      sha256=$hash"
            return
        }
        Remove-Item -LiteralPath $destination -Force
    }

    if ($VerifyOnly) {
        throw "Missing artifact: $($artifact.RelativePath)"
    }

    Write-Host "GET   $($artifact.Name)"
    Write-Host "      $($artifact.Url)"
    Write-Host "      -> $($artifact.RelativePath)"

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $curl) {
        & $curl.Source '--fail' '--location' '--retry' '5' '--retry-delay' '2' '--continue-at' '-' '--output' $partial $artifact.Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed with exit code $LASTEXITCODE for $($artifact.Url). Partial file kept at $partial"
        }
    } else {
        Invoke-WebRequest -UseBasicParsing -Uri $artifact.Url -OutFile $partial
    }

    if (-not (Test-Path -LiteralPath $partial -PathType Leaf)) {
        throw "Download did not create $partial"
    }

    Move-Item -LiteralPath $partial -Destination $destination -Force
    $hash = Get-Sha256 $destination
    Write-Host "DONE  $($artifact.RelativePath)"
    Write-Host "      sha256=$hash"
    Write-Host "      source=$($artifact.Source)"
}

$failed = 0
foreach ($artifact in $artifacts) {
    try {
        Download-Artifact $artifact
    } catch {
        $failed++
        Write-Error $_
    }
}

if ($failed -gt 0) {
    throw "$failed model artifact(s) failed. Re-run the command after fixing the reported error(s)."
}

Write-Host ''
if ($VerifyOnly) {
    Write-Host 'Model verification completed.'
} else {
    Write-Host 'Model setup completed. Keep the printed SHA-256 values in the deployment record.'
}


