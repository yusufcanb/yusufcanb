param(
    [Parameter(Mandatory=$true)]
    [string]$ArticleName
)

# Get current year and month
$year = Get-Date -Format 'yyyy'
$month = Get-Date -Format 'MM'

# Create directory path
$dir = "Articles\$year\$month"

# Create directory if it doesn't exist
if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "Created directory: $dir"
}

# Create the markdown file
$filePath = "$dir\$ArticleName.md"
New-Item -ItemType File -Path $filePath -Force | Out-Null

Write-Host "Created article: $filePath" -ForegroundColor Green
