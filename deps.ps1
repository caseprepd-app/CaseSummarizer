param([string]$target = "src", [int]$depth = 2)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outfile = "deps_${timestamp}.svg"
& ".\.venv\Scripts\python.exe" -m pydeps $target --max-bacon $depth -o $outfile --noshow
Write-Host "Generated $outfile"