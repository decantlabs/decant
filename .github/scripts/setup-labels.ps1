param([Parameter(Mandatory)][string]$Repo)
. "$PSScriptRoot/common.ps1"
Start-Setup $Repo 'labels'
$labels = Get-Content -Raw -Encoding UTF8 (Join-Path $Root '.github/labels.yml') | ConvertFrom-Json
foreach ($label in $labels) {
    Invoke-Gh @('label','create',$label.name,'--repo',$Repo,'--color',$label.color,'--description',$label.description,'--force')
}
