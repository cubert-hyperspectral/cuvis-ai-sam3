; Inno Setup script for CuvisAI SAM3 Server
; Requires Inno Setup 6+

#define MyAppName "CuvisAI SAM3 Server"
#define MyAppExeName "sam3-tray.exe"
#define MyServerExeName "sam3-rest-api.exe"
#define MyAppPublisher "Cubert GmbH"
#define MyAppURL "https://github.com/cubert-hyperspectral/cuvis-ai-sam3"

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

[Setup]
AppId={{D2A17747-C47A-45F3-97A3-BE52AA411661}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\CuvisAI\SAM3Server
DefaultGroupName=CuvisAI\SAM3 Server
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=CuvisAI-SAM3-Server-{#MyAppVersion}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
SetupIconFile=app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Dirs]
Name: "{app}\models"
Name: "{app}\configs"
Name: "{app}\logs"

[Files]
Source: "..\dist\sam3-server\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dist\sam3-tray\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dist\download-weights\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\configs\sam3-server.env"; DestDir: "{app}\configs"; Flags: ignoreversion

[Tasks]
Name: "downloadweights"; Description: "Download required model weights (~1-2 GB)"; Flags: checked
Name: "startattray"; Description: "Start server tray at Windows login"; Flags: checked

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "CuvisAI SAM3 Tray"; ValueData: """{app}\sam3-tray.exe"""; Tasks: startattray; Flags: uninsdeletevalue

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\SAM3 REST API (Console)"; Filename: "{app}\{#MyServerExeName}"
Name: "{group}\Download SAM3 Weights"; Filename: "{app}\download-weights.exe"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\sam3-tray.exe"; Description: "Launch SAM3 server tray"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  DownloadExe: string;
  Params: string;
begin
  if CurStep = ssPostInstall then
  begin
    if WizardIsTaskSelected('downloadweights') then
    begin
      DownloadExe := ExpandConstant('{app}\download-weights.exe');
      Params := '--path "' + ExpandConstant('{app}\models\sam3.pt') + '"';

      if not Exec(DownloadExe, Params, '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
      begin
        MsgBox('Could not start model weight downloader. Setup will abort.', mbCriticalError, MB_OK);
        Abort;
      end;

      if ResultCode <> 0 then
      begin
        MsgBox('Model weight download failed. Setup will abort.', mbCriticalError, MB_OK);
        Abort;
      end;
    end;
  end;
end;

