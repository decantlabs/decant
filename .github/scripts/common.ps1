Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
function Start-Setup([string]$Repo, [string]$Action) {
    if ($Repo -notmatch '^[\w.-]+/[\w.-]+$') { throw 'Expected OWNER/REPO' }
    $logDir = Join-Path $Root 'build/setup'
    New-Item -ItemType Directory -Force $logDir | Out-Null
    $script:Log = Join-Path $logDir ($Action + '.log')
    Get-Command gh -ErrorAction Stop | Out-Null
    Invoke-Gh @('auth','status') | Out-Null
    $target = ((Invoke-Gh @('repo','view',$Repo,'--json','visibility,nameWithOwner')) -join '') | ConvertFrom-Json
    if ($target.visibility -ne 'PRIVATE' -or $target.nameWithOwner -ne $Repo) { throw 'Expected the specified private repository.' }
    if ((Read-Host "Type $Repo to approve $Action on that repository") -cne $Repo) { throw 'Cancelled.' }
}
function Invoke-Gh([string[]]$Arguments) {
    Add-Content $script:Log ("{0:s} gh {1}" -f (Get-Date), ($Arguments[0..1] -join ' '))
    $output = & gh @Arguments
    if ($LASTEXITCODE -ne 0) { throw "gh failed ($LASTEXITCODE). Inspect authentication, repository role and project scopes. See the preceding CLI error." }
    return $output
}
function Read-Drafts {
    foreach ($file in Get-ChildItem (Join-Path $Root '.github/issue-drafts/*.md') | Sort-Object Name) {
        $body = Get-Content -Encoding UTF8 -Raw $file.FullName
        $title = [regex]::Match($body,'(?m)^# (.+)$').Groups[1].Value.Trim()
        $labels = [regex]::Match($body,'(?m)^Labels: (.+)$').Groups[1].Value.Trim() -split ' \| '
        if (!$title -or !$labels[0]) { throw "Invalid draft $($file.Name)" }
        [pscustomobject]@{Title=$title; File=$file.FullName; Labels=$labels}
    }
}
function Read-Issues([string]$Repo) {
    foreach ($row in Invoke-Gh @('api',"repos/$Repo/issues?state=all&per_page=100",'--paginate','--jq','.[] | select(.pull_request == null) | [.title,.html_url] | @tsv')) {
        $pair = $row -split "\t",2
        [pscustomobject]@{Title=$pair[0]; Url=$pair[1]}
    }
}
