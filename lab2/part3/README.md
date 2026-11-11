# Part 3：AdaptQuant 實驗

## 環境需求

- Python 3.11、支援 CUDA 的 GPU，以及可用的 NVIDIA 驅動程式。
- 可下載 `Qwen/Qwen3-4B` 模型與 WikiText-2 資料集，或已有對應快取。
- 本目錄中的 `thresholds.json` 是已固定的 reference thresholds，供正式實驗使用。

`run.sh` 會先安裝 `requirements.txt` 中的套件，包括 `torch==2.5.1+cu121`，再執行實驗。

## 執行正式 experiment

從專案根目錄進入 Part 3，執行量化與 PPL 評估：

```bash
cd lab2/part3
bash run.sh --thresholds thresholds.json \
  --dtype float16 --seed 0 --metrics-json experiment_reproduced.json
```

命令使用 `Qwen/Qwen3-4B`、float16 與指定 seed，並使用 WikiText-2 test 評估。終端輸出 PPL，而`experiment_reproduced.json` 會保存 PPL、weight bits、activation bits、realized effective bit rate 及 `passes_limit`。

## GPU 排程與 log

額外入口會等待可用 GPU，再透過原本的 `run.sh` 執行實驗：

```bash
python gpu_run.py -- --thresholds thresholds.json --dtype float16 --seed 0
# 已安裝依賴時
SKIP_INSTALL=1 python gpu_run.py -- --thresholds thresholds.json
```

runner 使用 Python 標準函式庫，需求為 Linux、`bash` 和 `nvidia-smi`。預設使用啟動 runner 的 Python，也支援 `PYTHON=/path/to/python`。依賴在排程前安裝一次，輸出保存至 `setup.log`。

每次啟動前查詢各卡的 total／used／free memory，選擇 free memory 最多且嚴格大於 **12288 MiB（12 GiB）** 的 GPU。取得檔案鎖後再次查詢，並以 GPU UUID 設定 `CUDA_VISIBLE_DEVICES`，每次實驗只使用一張卡。既有 `CUDA_VISIBLE_DEVICES` 的限制也會保留（支援實體 index 或唯一 UUID 前綴）。

沒有符合條件的 GPU 或可取得的鎖時，每 30 秒重查，等待過程不寫入 log；可用 `--poll-seconds` 調整間隔。Part2、Part3 共用 `/tmp/edgeai-gpu-<UID>/<GPU UUID>.lock`，避免同一使用者的 runner 重複派發到同一張卡。鎖不會限制其他程式；符合門檻後仍可能在實驗中遇到 OOM。

每次執行會顯示獨立的 `results/gpu_runs/<UTC 時間戳記>_<PID>/`，保存 `setup.log`、`runner.log`、`experiment.log`、`experiment.json` 和 `summary.json`。實驗 stdout／stderr 合併至 `experiment.log`；`experiment.json` 沿用完整 metrics，包括 PPL、weight／activation bits、realized effective bit rate、`passes_limit`、格式分配與儲存資訊。摘要另記錄初始 GPU 狀態、啟動前記憶體、GPU UUID、命令、環境設定、時間、耗時、退出碼與完整 metrics。

runner 管理 metrics 路徑，請勿另傳 `--metrics-json`。實驗失敗、metrics 缺失／無法解析或 GPU 查詢失敗時，以非零退出碼結束並保存錯誤。Ctrl+C 或 SIGTERM 會先終止實驗子程序，再釋放鎖。單次 Part3 實驗只使用一張卡；另一張卡可同時供 Part2 或另一個 runner 使用。


## Appendix：產生 thresholds

現有 `thresholds.json` 可直接用於正式實驗。若需重新推導，需取得 Hugging Face 上 `meta-llama/Llama-3.1-8B` 的存取權，以及支援 bfloat16 的 CUDA GPU。安裝本專案依賴後，在 `lab2/part3` 執行：

```bash
python derive_thresholds.py --target-bits 5.3 --output thresholds_new.json
```

程式預設使用 WikiText-2 train 的 2048 個 tokens，產生 `thresholds_new.json` 與 `thresholds_new.tokens.json`。範例使用不同檔名，不會覆寫正式實驗使用的 `thresholds.json`。
