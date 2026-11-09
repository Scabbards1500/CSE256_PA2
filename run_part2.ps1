param(
    [string]$Python = 'D:\DevTools\Anaconda3-5.2.0\envs\pytorch311\python.exe'
)
$previousPythonPath = $env:PYTHONPATH
try {
    $packagePath = Join-Path $PSScriptRoot '.local_packages'
    if (Test-Path -LiteralPath $packagePath) {
        $env:PYTHONPATH = if ($previousPythonPath) { "$packagePath;$previousPythonPath" } else { $packagePath }
    }
    & $Python (Join-Path $PSScriptRoot 'main.py') part2
    if ($LASTEXITCODE -ne 0) { throw "Part 2 exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
