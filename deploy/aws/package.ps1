param(
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location $packageRoot
try {
    & $Python (Join-Path $PSScriptRoot 'package_release.py') --out $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Release packaging failed' }
} finally { Pop-Location }
