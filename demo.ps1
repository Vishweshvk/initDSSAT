param(
    [string]$Experiment = "KSAS8101",
    [int]$Treatment = 1,
    [double]$Area = 2,
    [ValidateSet("hectares", "acres")]
    [string]$AreaUnit = "acres"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$projectPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $projectPython) {
    $python = $projectPython
} else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

function Invoke-ProjectCommand {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    Write-Host "`n> $python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE. Fix the reported issue before continuing."
    }
}

Invoke-ProjectCommand @("-m", "cli.main", "doctor")
Invoke-ProjectCommand @("-m", "cli.main", "list-experiments", "--crop", "wheat")
Invoke-ProjectCommand @("-m", "cli.main", "validate-experiment", "--experiment", $Experiment)
Invoke-ProjectCommand @("-m", "cli.main", "evaluate", "--crop", "wheat")

Write-Host "`nStarting the local web app. Open http://127.0.0.1:5000 and enter $Experiment / treatment $Treatment / $Area $AreaUnit." -ForegroundColor Green
Write-Host "Press Ctrl+C here to stop the app." -ForegroundColor DarkGray
& $python "app.py"
