# Setup WSL and Build C++ Worker
# Run this AFTER WSL is installed and you've restarted

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "WSL Setup & C++ Worker Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if WSL is installed
Write-Host "Checking WSL installation..." -ForegroundColor Green
$wslStatus = wsl --status 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: WSL is not installed!" -ForegroundColor Red
    Write-Host "Please run install_wsl.ps1 first as Administrator" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "WSL is installed!" -ForegroundColor Green
Write-Host ""

# Get the project path in WSL format
$windowsPath = (Get-Location).Path
$wslPath = $windowsPath -replace '^([A-Z]):', { '/mnt/' + $_.Groups[1].Value.ToLower() } -replace '\\', '/'

Write-Host "Project path (Windows): $windowsPath" -ForegroundColor Cyan
Write-Host "Project path (WSL): $wslPath" -ForegroundColor Cyan
Write-Host ""

# Create a setup script for WSL
$wslSetupScript = @"
#!/bin/bash
echo "========================================="
echo "Installing dependencies in WSL..."
echo "========================================="
echo ""

# Update package list
echo "Updating package list..."
sudo apt update

# Install dependencies
echo ""
echo "Installing g++, liblzma-dev, libxxhash-dev..."
sudo apt install -y g++ liblzma-dev libxxhash-dev build-essential

echo ""
echo "========================================="
echo "Building C++ NCD Worker..."
echo "========================================="
echo ""

# Navigate to project directory
cd "$wslPath"

# Build the C++ worker
bash build.sh

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "You can now run: python fast_cluster.py"
echo ""
"@

# Save the script to a temporary file
$tempScript = Join-Path $env:TEMP "wsl_setup.sh"
$wslSetupScript | Out-File -FilePath $tempScript -Encoding UTF8

# Copy script to WSL and execute
Write-Host "Running setup in WSL..." -ForegroundColor Green
Write-Host ""

wsl bash -c "cat > /tmp/wsl_setup.sh << 'EOF'
$wslSetupScript
EOF
chmod +x /tmp/wsl_setup.sh
/tmp/wsl_setup.sh
"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "All Done!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the clustering:" -ForegroundColor Cyan
Write-Host "  wsl" -ForegroundColor White
Write-Host "  cd $wslPath" -ForegroundColor White
Write-Host "  python fast_cluster.py" -ForegroundColor White
Write-Host ""
