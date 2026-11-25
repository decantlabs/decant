param([Parameter(Mandatory)][string]$Repo)
. "$PSScriptRoot/common.ps1"
Start-Setup $Repo 'project'
$owner = $Repo.Split('/')[0]
$title = 'Decant'
function Project-List([string]$command,[string[]]$extra,[string]$key) {
    $parameters = @('project',$command) + $extra + @('--owner',$owner,'--limit','1000','--format','json')
    $result = ((Invoke-Gh $parameters) -join '') | ConvertFrom-Json
    if ($result.totalCount -gt 1000) { throw 'Pagination required before making changes.' }
    return $result.$key
}
$all = @(Project-List 'list' @() 'projects') + @(Project-List 'list' @('--closed') 'projects')
$matchesFound = @($all | Where-Object title -ceq $title | Sort-Object number -Unique)
if ($matchesFound.Count -gt 1) { throw 'Ambiguous project title.' }
if (!$matchesFound.Count) {
    $project = ((Invoke-Gh @('project','create','--owner',$owner,'--title',$title,'--format','json')) -join '') | ConvertFrom-Json
} else { $project = $matchesFound[0] }
if ($project.PSObject.Properties['closed'] -and $project.closed) { throw 'Project closed. Review manually.' }
$num = [string]$project.number
$details = ((Invoke-Gh @('project','view',$num,'--owner',$owner,'--format','json')) -join '') | ConvertFrom-Json
if (!$details.PSObject.Properties['public'] -or $details.public -ne $false) { throw 'Confirm a private project before linking issues.' }
Invoke-Gh @('project','link',$num,'--owner',$owner,'--repo',$Repo)
$fields = @(Project-List 'field-list' @($num) 'fields')
$options = [ordered]@{
    Priority='critical,high,medium,low'
    Component='decant,d3d12x,shaders,memory,dxr,upscaling,textures,diagnostics,documentation'
    Difficulty='beginner,intermediate,advanced,expert'
    'Validation Status'='Unmeasured,Needs testing,Verified,Failed'
    'Executable Dependency'='Independent,Required'
}
foreach ($name in $options.Keys) {
    if (!($fields | Where-Object name -eq $name)) {
        Invoke-Gh @('project','field-create',$num,'--owner',$owner,'--name',$name,'--data-type','SINGLE_SELECT','--single-select-options',$options[$name])
    }
}
$items = @(Project-List 'item-list' @($num) 'items')
$urls = @($items | ForEach-Object { if ($_.content.PSObject.Properties['url']) { $_.content.url } })
$drafts = @(Read-Drafts)
foreach ($issue in Read-Issues $Repo) {
    if (($drafts.Title -contains $issue.Title) -and ($urls -notcontains $issue.Url)) {
        Invoke-Gh @('project','item-add',$num,'--owner',$owner,'--url',$issue.Url)
    }
}
Write-Host 'Complete status columns, views, privacy, and field assignments using docs/GITHUB_PROJECT_SETUP.md.'
