# watchdog.ps1 - self-healing supervisor for the NAS100 downloader.
# Relaunches run.py if it dies before the run completes (final_report.json).
# Single-instance guarded; stops with NEEDS_ATTENTION.txt if relaunching makes
# no progress (avoids a useless loop).
param([string]$Root = "D:\NAS 100 Data")
$py = Join-Path $Root ".venv\Scripts\python.exe"
$runpy = Join-Path $Root "scripts\run.py"
$pidFile = Join-Path $Root "logs\downloader.pid"
$logOut = Join-Path $Root "logs\console.out.log"
$logErr = Join-Path $Root "logs\console.err.log"
$wdLog = Join-Path $Root "logs\watchdog.log"
$attn = Join-Path $Root "NEEDS_ATTENTION.txt"
$statePath = Join-Path $Root "download_state.json"
$finalReport = Join-Path $Root "reports\final_report.json"

function WLog($m) { "$(Get-Date -Format o) | $m" | Add-Content $wdLog }
function CompletedCount {
    if (-not (Test-Path $statePath)) { return 0 }
    try { $s = Get-Content $statePath -Raw -EA Stop | ConvertFrom-Json } catch { return 0 }
    $n = 0
    foreach ($p in $s.months.PSObject.Properties) {
        if ($p.Value.status -in @('completed', 'no_data')) { $n++ }
    }
    return $n
}

# single-instance guard
$selfPid = Join-Path $Root "logs\watchdog.process.pid"
if (Test-Path $selfPid) {
    $e = (Get-Content $selfPid -EA SilentlyContinue | Select-Object -First 1)
    if ($e) { $e = "$e".Trim() }
    if ($e -and $e -ne "$PID" -and (Get-Process -Id ([int]$e) -EA SilentlyContinue)) {
        WLog "another watchdog ($e) already running; exiting ($PID)"; exit 0
    }
}
"$PID" | Out-File $selfPid -Encoding ascii

WLog "watchdog started"
$lastProgress = CompletedCount; $strikes = 0; $relaunches = 0
while ($true) {
    Start-Sleep -Seconds 300
    if (Test-Path $finalReport) { WLog "final_report.json present -> complete; exiting"; break }
    $dlpid = $null
    if (Test-Path $pidFile) { $dlpid = (Get-Content $pidFile | Select-Object -First 1).Trim() }
    $alive = $false
    if ($dlpid) {
        $proc = Get-Process -Id ([int]$dlpid) -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -match 'python') { $alive = $true }
    }
    if ($alive) {
        $now = CompletedCount
        if ($now -gt $lastProgress) { $lastProgress = $now; $strikes = 0 }
        continue
    }
    $now = CompletedCount
    if ($relaunches -gt 0 -and $now -le $lastProgress) {
        $strikes++; WLog "no progress since relaunch (completed=$now); strike $strikes"
        if ($strikes -ge 3) {
            "NAS100 downloader stalled at $now months completed. Check logs\download.log and logs\errors.log, then run run.bat." | Set-Content $attn
            WLog "3 strikes -> wrote NEEDS_ATTENTION.txt; exiting"; break
        }
    } else { $strikes = 0 }
    $lastProgress = $now
    if ($relaunches -ge 20) { "Relaunch cap hit at $now months." | Set-Content $attn; WLog "relaunch cap; exiting"; break }
    WLog "downloader dead, run incomplete (completed=$now) -> relaunching"
    $p = Start-Process -FilePath $py -ArgumentList "`"$runpy`"" -WorkingDirectory $Root `
         -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru
    $p.Id | Out-File $pidFile -Encoding ascii
    $relaunches++; WLog "relaunched PID $($p.Id) (#$relaunches)"
}
