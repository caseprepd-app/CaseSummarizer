; ──────────────────────────────────────────────────────────────
; CasePrepd Installer - Inno Setup Script
; Packages the PyInstaller output into a Windows installer
; with desktop/start menu shortcuts and uninstaller.
; ──────────────────────────────────────────────────────────────

#define MyAppName "CasePrepd"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "CasePrepd"
#define MyAppURL "https://sites.google.com/view/caseprepd/home"
#define MyAppExeName "CasePrepd.exe"

[Setup]
AppId={{B8F3A1D2-7E4C-4B9A-A6D5-3F2E1C8B9D4A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
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
; Show estimated install size in Add/Remove Programs
UninstallDisplayName={#MyAppName} {#MyAppVersion}
VersionInfoVersion={#MyAppVersion}.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=100% Offline Legal Document Processor for Court Reporters
VersionInfoProductName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Bundle everything from the PyInstaller dist output
Source: "..\dist\CasePrepd\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\_internal\assets\icon.ico"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\_internal\assets\icon.ico"; Tasks: desktopicon

[UninstallDelete]
; Clean up files created at runtime inside the install directory
Type: filesandordirs; Name: "{app}\*"
; Clean up user data in %APPDATA%\CasePrepd (logs, cache, preferences, feedback, corpus)
Type: filesandordirs; Name: "{userappdata}\{#MyAppName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
