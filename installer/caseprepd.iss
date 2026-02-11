; ──────────────────────────────────────────────────────────────
; CasePrepd Installer - Inno Setup Script
; Packages the PyInstaller output into a Windows installer
; with desktop/start menu shortcuts and uninstaller.
; ──────────────────────────────────────────────────────────────

#define MyAppName "CasePrepd"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "CasePrepd"
#define MyAppExeName "CasePrepd.exe"

[Setup]
AppId={{B8F3A1D2-7E4C-4B9A-A6D5-3F2E1C8B9D4A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=CasePrepdSetup
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
DiskSpanning=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Bundle everything from the PyInstaller dist output
Source: "..\dist\CasePrepd\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\icon.ico"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\icon.ico"; Tasks: desktopicon

[UninstallDelete]
; Clean up any files created at runtime (Python __pycache__, logs, temp files)
Type: filesandordirs; Name: "{app}\*"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
