; Inno Setup script — per-user install, no admin. Invoked by build_windows.ps1 with /DMyAppVersion and /DDistDir.
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef DistDir
  #define DistDir "..\..\dist"
#endif
[Setup]
AppName=Atelier
AppVersion={#MyAppVersion}
AppPublisher=Atelier
DefaultDirName={localappdata}\Programs\Atelier
DefaultGroupName=Atelier
PrivilegesRequired=lowest
OutputDir={#DistDir}
OutputBaseFilename=Atelier-windows-x64-setup
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\icon.ico
UninstallDisplayIcon={app}\Atelier.exe
[Files]
Source: "{#DistDir}\Atelier\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
[Icons]
Name: "{group}\Atelier"; Filename: "{app}\Atelier.exe"
Name: "{userdesktop}\Atelier"; Filename: "{app}\Atelier.exe"; Tasks: desktopicon
[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"
[Run]
Filename: "{app}\Atelier.exe"; Description: "Launch Atelier"; Flags: nowait postinstall skipifsilent
