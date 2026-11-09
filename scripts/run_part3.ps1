param(
    [string]$Python = (Join-Path (Split-Path -Parent $PSScriptRoot) '.part3_env\Scripts\python.exe')
)
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Set up the Part 3 environment first; see README.md.'
}
Push-Location $projectRoot
try {
    & $Python scripts\notebook\execute_part3_notebook.py
    if ($LASTEXITCODE -ne 0) { throw 'Part 3 notebook execution failed.' }
    & $Python scripts\notebook\export_part3_notebook.py
    if ($LASTEXITCODE -ne 0) { throw 'Part 3 notebook PDF export failed.' }
} finally {
    Pop-Location
}
