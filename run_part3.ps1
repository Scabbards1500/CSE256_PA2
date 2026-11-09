param(
    [string]$Python = (Join-Path $PSScriptRoot '.part3_env\Scripts\python.exe')
)
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Set up the Part 3 environment first; see README.txt.'
}
Push-Location $PSScriptRoot
try {
    & $Python scripts\execute_part3_notebook.py
    if ($LASTEXITCODE -ne 0) { throw 'Part 3 notebook execution failed.' }
    & $Python scripts\export_part3_notebook.py
    if ($LASTEXITCODE -ne 0) { throw 'Part 3 notebook PDF export failed.' }
} finally {
    Pop-Location
}
