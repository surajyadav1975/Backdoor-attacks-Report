# Run the full PSBD pipeline against all three attacks (Windows / PowerShell).
# Use from the project root:
#   .\scripts\run_all_attacks.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> BadNets"
python main.py --attack badnets

Write-Host "==> Blended"
python main.py --attack blended

Write-Host "==> WaNets"
python main.py --attack wanets

Write-Host "All experiments finished. See ./results for JSON reports."
