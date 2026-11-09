param(
    [string]$Python = 'D:\DevTools\Anaconda3-5.2.0\envs\pytorch311\python.exe'
)
$projectRoot = Split-Path -Parent $PSScriptRoot
# This machine has NLTK in Anaconda base; a workspace copy avoids changing envs.
$previousPythonPath = $env:PYTHONPATH
try {
    $packagePath = Join-Path $projectRoot '.local_packages'
    if (Test-Path -LiteralPath $packagePath) {
        $env:PYTHONPATH = if ($previousPythonPath) { "$packagePath;$previousPythonPath" } else { $packagePath }
    }
    & $Python (Join-Path $projectRoot 'main.py') part1
    if ($LASTEXITCODE -ne 0) { throw "Part 1 exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
