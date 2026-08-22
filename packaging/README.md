# Packaging

One PyInstaller spec (`atelier.spec`), one build script per OS, one headless smoke test. CI
(`.github/workflows/desktop-builds.yml`) runs the four legs and, on a `v*` tag, publishes a release.

    packaging/build_macos.sh                                     # dist/Atelier.app, -mac-<arch>.zip, .dmg
    powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1   # dist\Atelier\Atelier.exe, .zip, setup.exe (Inno Setup)
    packaging/build_linux.sh                                     # dist/Atelier/, -linux-<arch>.tar.gz
    packaging/smoke_test.py "<binary or 'python atelier_app.py'>" [--frozen]

The builds are **unsigned**. macOS: the first launch needs right-click → Open (Gatekeeper); if it still refuses,
`xattr -dr com.apple.quarantine /Applications/Atelier.app`. Windows: SmartScreen → "More info → Run anyway".
Signing/notarising needs an Apple Developer account / a code-signing certificate and is not part of this repo.

The bundles do not contain the pretrained weights (VGG19 ~550 MB, SD-Turbo ~2.5 GB); the app downloads them
once into its sandbox folder (`~/Library/Application Support/Atelier`, `%LOCALAPPDATA%\Atelier`,
`~/.local/share/atelier`) and never again. Bundle size is roughly 1–1.5 GB because of torch.
