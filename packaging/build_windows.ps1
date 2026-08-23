# Build Atelier on Windows: icon -> PyInstaller -> dist\Atelier\Atelier.exe -> dist\Atelier-windows-x64.zip (+ Inno Setup installer if ISCC is present) -> smoke test.
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path; $Root = Split-Path -Parent $Here; $Dist = Join-Path $Root "dist"
$Py = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) { Join-Path $Root ".venv\Scripts\python.exe" } else { "python" }
& $Py -c "import PyInstaller"; if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt -r requirements-desktop.txt first" }
$m = Select-String -Path (Join-Path $Root "atelier_app.py") -Pattern '^__version__\s*=\s*["'']([^"'']+)["'']' | Select-Object -First 1
$Version = if ($m) { $m.Matches[0].Groups[1].Value } else { "0.0.0" }
Write-Host "==> Atelier $Version - Windows x64 - $Py"
if (-not (Test-Path (Join-Path $Here "icon.ico"))) { & $Py (Join-Path $Here "make_icon.py") --ico-only }
if (Test-Path (Join-Path $Dist "Atelier")) { Remove-Item -Recurse -Force (Join-Path $Dist "Atelier") }
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
Push-Location $Root
try { & $Py -m PyInstaller --noconfirm --distpath $Dist --workpath (Join-Path $Here "build") (Join-Path $Here "atelier.spec"); if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" } }
finally { Pop-Location }
$Zip = Join-Path $Dist "Atelier-windows-x64.zip"; if (Test-Path $Zip) { Remove-Item $Zip }
Compress-Archive -Path (Join-Path $Dist "Atelier") -DestinationPath $Zip
$Iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $Iscc) { $cands = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe"); foreach ($c in $cands) { if (Test-Path $c) { $Iscc = @{ Source = $c }; break } } }
if ($Iscc -and -not $env:SKIP_INSTALLER) {
  $Iss = Join-Path $Here "windows\atelier.iss"
  & $Iscc.Source "/DMyAppVersion=$Version" "/DDistDir=$Dist" $Iss; if ($LASTEXITCODE -ne 0) { Write-Warning "Inno Setup failed; zip only" }
} else { Write-Warning "Inno Setup (ISCC.exe) not found - zip only" }
if (-not $env:SKIP_SMOKE) { & $Py (Join-Path $Here "smoke_test.py") (Join-Path $Dist "Atelier\Atelier.exe") --frozen --deep; if ($LASTEXITCODE -ne 0) { throw "smoke test failed" } }
Get-ChildItem $Dist | Where-Object { $_.Name -like "Atelier-windows*" } | Format-Table Name, Length
