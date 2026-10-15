# CPU timings on a laptop (Intel i7-1185G7, 4 cores / 8 threads), mains power, machine idle.
# Reuses the SciFact index and teacher logits built on the GPU machine (research/cache-laptop).
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root "..\src-tauri\target-bench\release\examples\retrieval_eval.exe"
$data = Join-Path $root "data"
$runs = Join-Path $root "runs\latency-laptop"
$cache = Join-Path $root "cache-laptop"
$models = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $runs | Out-Null

function Run($out, [string[]]$extra) {
    $log = Join-Path $runs ($out.Replace('/', '_') + ".log")
    & cmd /c "`"$exe`" --data `"$data\$($out.Split('/')[1])`" --out `"$runs\$out`" --cache `"$cache`" $($extra -join ' ') > `"$log`" 2>&1"
    Write-Output "done $out"
}

Run "query/scifact" @("--model", "multilingual-e5-small", "--timing-queries", "10", "--depths", "10,20,90")
foreach ($s in @("rr-L12-mse", "rr-L6-mse")) {
    foreach ($v in @("fp32", "int8")) {
        Run "$s-$v/scifact" @("--model", "multilingual-e5-small", "--reranker", "$models\$s-onnx\$v", "--timing-queries", "20", "--depths", "20")
    }
}
foreach ($m in @(@("index-e5-small", "multilingual-e5-small"), @("index-both-fp32", "$models\e5-small-both-onnx\fp32"), @("index-both-int8", "$models\e5-small-both-onnx\int8"))) {
    $log = Join-Path $runs "$($m[0])_syntec.log"
    & cmd /c "`"$exe`" --model `"$($m[1])`" --data `"$data\syntec`" --out `"$runs\$($m[0])\syntec`" --cache `"$cache-index`" --rerank false --timing-queries 100 > `"$log`" 2>&1"
    Write-Output "done $($m[0])"
}
