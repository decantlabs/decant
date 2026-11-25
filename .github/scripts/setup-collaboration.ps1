param([Parameter(Mandatory)][string]$Repo)
. "$PSScriptRoot/common.ps1"
Start-Setup $Repo 'collaboration'
foreach ($name in @('setup-labels','create-issues','setup-project')) {
    & "$PSScriptRoot/$name.ps1" -Repo $Repo
}
