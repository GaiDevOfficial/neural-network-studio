[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Tag,
    [string]$Version = "",
    [string]$NotesFile = "",
    [string]$Owner = "GaiDevOfficial",
    [string]$Repo = "neural-network-studio"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Version) { $Version = $Tag -replace '^v', '' }
if (-not $NotesFile) {
    $candidate = Join-Path $root "release-notes\$Tag.md"
    if (Test-Path $candidate) { $NotesFile = $candidate }
}

if ($NotesFile -and (Test-Path $NotesFile)) {
    $notes = Get-Content -Raw -LiteralPath $NotesFile
} else {
    $previousTag = ""
    try { $previousTag = (git describe --tags --abbrev=0 2>$null).Trim() } catch { }
    $range = if ($previousTag) { "$previousTag..HEAD" } else { "HEAD" }
    $changes = (git log --pretty=format:'- %s (`%h`)' $range) -join "`n"
    $notes = "## What's changed`n`n$changes"
}

$zip = Join-Path $root "dist\NeuralNetworkStudio-Windows-x64.zip"
$checksumFile = Join-Path $root "dist\NeuralNetworkStudio-Windows-x64.sha256.txt"
if (-not (Test-Path $zip)) { throw "Release asset missing: $zip. Run python desktop/build_exe.py first." }
if (-not (Test-Path $checksumFile)) { throw "Checksum missing: $checksumFile. Run python desktop/build_exe.py first." }

$checksum = (Get-Content -Raw -LiteralPath $checksumFile).Trim()
$notes += "`n`n### SHA-256`n`n``````text`n$checksum`n```````n"
$notes += "`n**Full changelog:** [CHANGELOG.md](https://github.com/$Owner/$Repo/blob/main/CHANGELOG.md)`n"

function Get-GitHubToken {
    $previousPrompt = $env:GIT_TERMINAL_PROMPT
    try {
        $env:GIT_TERMINAL_PROMPT = "0"
        $credentialInput = "protocol=https`nhost=github.com`n`n"
        $credentialOutput = $credentialInput | git credential fill 2>$null
        $token = (($credentialOutput -split "`n") | Where-Object { $_ -like 'password=*' } | Select-Object -First 1) -replace '^password=', ''
        if (-not $token) { throw "No GitHub credential found. Sign in with Git Credential Manager first." }
        return $token
    } finally {
        $env:GIT_TERMINAL_PROMPT = $previousPrompt
    }
}

$token = Get-GitHubToken
$headers = @{
    Authorization = "Bearer $token"
    Accept = "application/vnd.github+json"
    "User-Agent" = "node-studio-release"
    "X-GitHub-Api-Version" = "2022-11-28"
}
$api = "https://api.github.com/repos/$Owner/$Repo"

$release = $null
try {
    $release = Invoke-RestMethod -Uri "$api/releases/tags/$Tag" -Headers $headers -Method Get
    Write-Host "Release $Tag already exists; ensuring assets are uploaded."
} catch {
    $payload = @{
        tag_name = $Tag
        target_commitish = "main"
        name = "Node Studio $Version"
        body = $notes
        draft = $false
        prerelease = $false
        generate_release_notes = $false
    } | ConvertTo-Json -Depth 6
    $release = Invoke-RestMethod -Uri "$api/releases" -Headers $headers -Method Post -Body $payload -ContentType "application/json"
    Write-Host "Created release: $($release.html_url)"
}

$existingAssets = @()
try {
    $existingAssets = Invoke-RestMethod -Uri "$api/releases/$($release.id)/assets" -Headers $headers -Method Get
} catch { }

function Publish-Asset {
    param([string]$Path)
    $name = Split-Path $Path -Leaf
    if ($existingAssets.name -contains $name) {
        Write-Host "Asset already present: $name"
        return
    }
    $encoded = [uri]::EscapeDataString($name)
    $uploadUrl = "https://uploads.github.com/repos/$Owner/$Repo/releases/$($release.id)/assets?name=$encoded"
    $result = Invoke-RestMethod -Uri $uploadUrl -Headers $headers -Method Post -ContentType "application/octet-stream" -InFile $Path
    Write-Host "Uploaded asset: $($result.name) ($([math]::Round($result.size / 1MB, 2)) MB)"
}

Publish-Asset $zip
Publish-Asset $checksumFile

Write-Host ""
Write-Host "Release published: $($release.html_url)"
Write-Host "Tag             : $Tag"
Write-Host "ZIP             : $zip"
Write-Host "SHA-256         : $checksum"
