; ──────────────────────────────────────────────────────────────
; CasePrepd Installer - Inno Setup Script
; Packages the PyInstaller output into a Windows installer
; with desktop/start menu shortcuts and uninstaller.
; ──────────────────────────────────────────────────────────────

#define MyAppName "CasePrepd"
#define MyAppVersion "1.0.25"
#define MyAppPublisher "CasePrepd"
#define MyAppURL "https://caseprepd-app.github.io/CaseSummarizer/"
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
DiskSpanning=no
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
; VC++ Redistributable — installed silently if not already present
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

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

[Code]
// Install the bundled VC++ Redistributable silently if not already present.
// Native extensions (onnxruntime, tokenizers) depend on vcruntime140.dll.
function VCRuntimeInstalled: Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\vcruntime140.dll'));
end;

procedure InstallVCRedist;
var
  ResultCode: Integer;
begin
  if not VCRuntimeInstalled then
  begin
    Exec(ExpandConstant('{tmp}\vc_redist.x64.exe'),
         '/install /quiet /norestart', '', SW_HIDE,
         ewWaitUntilTerminated, ResultCode);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
    InstallVCRedist;
end;
