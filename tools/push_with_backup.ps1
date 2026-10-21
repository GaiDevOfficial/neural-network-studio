[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Message,
    [string]$Version = "",
    [switch]$SkipTests,
    [switch]$SkipBuild,
    [switch]$SkipPackagedSmoke,
    [switch]$SkipRelease
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".git")) { throw "Not a git repository: $root" }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "main") { throw "Releases must be created from main (current: $branch)." }

$previousTag = ""
try { $previousTag = (git describe --tags --abbrev=0 2>$null).Trim() } catch { }
$previousCommit = (git rev-parse HEAD).Trim()

if ($Version) {
    $newVersion = $Version -replace '^v', ''
} elseif ($previousTag -match '^v?(\d+)\.(\d+)\.(\d+)$') {
    $newVersion = "$($matches[1]).$($matches[2]).$([int]$matches[3] + 1)"
} else {
    $newVersion = "1.0.1"
}
$newTag = "v$newVersion"
if (git tag --list $newTag) { throw "Tag already exists: $newTag" }

# Keep package.json's version in step with the tag. The regex replaces only
# the top-level "version" value and preserves the rest of the file.
$packagePath = Join-Path $root "package.json"
$packageText = Get-Content -Raw -LiteralPath $packagePath
$versionPattern = '("version"\s*:\s*")[^"]+(")'
if ($packageText -notmatch $versionPattern) { throw "Could not find the version field in package.json." }
$updatedPackage = $packageText -replace $versionPattern, ('${1}' + $newVersion + '${2}')
if ($updatedPackage -ne $packageText) {
    [System.IO.File]::WriteAllText($packagePath, $updatedPackage, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "package.json version -> $newVersion"
}

Write-Host "Previous release: $previousTag"
Write-Host "New release     : $newTag"
Write-Host ""

# Keep one local copy of the previous released tree.
New-Item -ItemType Directory -Force -Path (Join-Path $root "backups") | Out-Null
$backupName = if ($previousTag) { "NeuralNetworkStudio-$previousTag.zip" } else { "NeuralNetworkStudio-previous.zip" }
$backupPath = Join-Path $root "backups\$backupName"
if (-not (Test-Path $backupPath)) {
    git archive --format=zip --output="$backupPath" $previousCommit
    if ($LASTEXITCODE -ne 0) { throw "Could not archive the previous version." }
}
Get-ChildItem (Join-Path $root "backups") -Filter "NeuralNetworkStudio-*.zip" |
    Where-Object { $_.FullName -ne (Resolve-Path $backupPath).Path } |
    Remove-Item -Force
Write-Host "Previous version backup: $backupPath"

if (-not $SkipTests) {
    Write-Host "`nRunning JavaScript suites..."
    npm test
    if ($LASTEXITCODE -ne 0) { throw "npm test failed; release aborted." }

    Write-Host "`nRunning Python suites..."
    npm run test:python
    if ($LASTEXITCODE -ne 0) { throw "npm run test:python failed; release aborted." }
}

if (-not $SkipBuild) {
    Write-Host "`nBuilding portable app and public ZIP..."
    python desktop/build_exe.py
    if ($LASTEXITCODE -ne 0) { throw "desktop/build_exe.py failed; release aborted." }

    if (-not $SkipPackagedSmoke) {
        Write-Host "`nRunning packaged WebView2 boot smoke..."
        node tools/smoke_packaged_exe.mjs
        if ($LASTEXITCODE -ne 0) { throw "Packaged smoke failed; release aborted." }
    }
}

git add -A
$staged = git diff --cached --name-only
if (-not $staged) { throw "No changes to commit." }

git commit -m $Message
if ($LASTEXITCODE -ne 0) { throw "git commit failed." }

git tag -a $newTag -m "Node Studio $newVersion"
if ($LASTEXITCODE -ne 0) { throw "git tag failed." }

if ($previousCommit) {
    git branch -f backup $previousCommit
    if ($LASTEXITCODE -ne 0) { throw "Could not update the backup branch." }
}

Write-Host "`nPushing main, backup and $newTag..."
git push origin backup --force
if ($LASTEXITCODE -ne 0) { throw "git push backup failed." }
git push origin main
if ($LASTEXITCODE -ne 0) { throw "git push main failed." }
git push origin $newTag
if ($LASTEXITCODE -ne 0) { throw "git push tag failed." }

if (-not $SkipRelease) {
    Write-Host "`nPublishing GitHub release..."
    & (Join-Path $PSScriptRoot "publish_github_release.ps1") -Tag $newTag -Version $newVersion
    if ($LASTEXITCODE -ne 0) { throw "GitHub release publishing failed." }
}

Write-Host ""
Write-Host "Released Node Studio $newVersion."
Write-Host "Backup branch points at: $previousCommit (one version behind)."
Write-Host "Local backup: $backupPath"
