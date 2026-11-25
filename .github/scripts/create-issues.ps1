param([Parameter(Mandatory)][string]$Repo)
. "$PSScriptRoot/common.ps1"
Start-Setup $Repo 'issues'
$existing = @(Read-Issues $Repo)
foreach ($draft in Read-Drafts) {
    if ($existing | Where-Object Title -eq $draft.Title) { continue }
    $arguments = @('issue','create','--repo',$Repo,'--title',$draft.Title,'--body-file',$draft.File)
    foreach ($label in $draft.Labels) { $arguments += @('--label',$label) }
    Invoke-Gh $arguments
}
