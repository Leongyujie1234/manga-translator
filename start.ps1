[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:OPENROUTER_KEY = "sk-or-v1-YOUR-KEY-HERE"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
Set-Location $PSScriptRoot
python app.py
