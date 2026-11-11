# Part 2：Qwen3-4B Linear 壓縮

程式壓縮 Qwen/Qwen3-4B decoder 中的 q/k/v/o 與 gate/up/down Linear，並以 WikiText-2 test 計算 PPL。目標層不包含 embedding 與 `lm_head`。四個 Case 都從原始模型開始載入。

## 環境

使用 Python 3.11、CUDA GPU 與 `requirements.txt` 指定的 PyTorch 2.5.1+cu121。執行環境需能下載 Hugging Face 模型、tokenizer 與 WikiText-2 資料集，或已準備相應快取。

```bash
cd lab2/part2
python -m pip install -r requirements.txt
```

## 執行

```bash
# 安裝依賴並依序執行四個 Case
bash 314552017_run.sh

# 依賴已安裝時跳過安裝
SKIP_INSTALL=1 bash 314552017_run.sh

# 單獨執行並保存詳細資料
python run_eval.py --case 3 --metrics-json results/case3.json
python run_eval.py --case 4 --metrics-json results/case4.json

# 原始模型
python run_eval.py
```

正式入口預設 Case 1、2 使用 GPTQ，Case 3 使用 whitening SVD 與 RTN，Case 4 使用 SparseGPT。`CASE1_QUANTIZER` 和 `CASE2_QUANTIZER` 可覆寫前兩個 Case 的量化方法。直接執行 `run_eval.py` 時，Case 1 預設為 RTN，Case 2 預設為 GPTQ。`--help` 列出位寬、group size、校正及 SVD 參數。

各 Case 顯示 Case、Compression Ratio 與 PPL；原始模型顯示 Base、Compression Ratio 和 PPL。此輸出格式同時適用正式入口和單獨評估；`--submission-output` 為相容參數。Case 3 的 SVD 壓縮率與 Case 4 的剪枝指標保存在 metrics JSON。

`--metrics-json` 保存設定、PPL、逐層儲存帳本及適用的壓縮診斷。正式入口自動寫入 `results/case1.json` 至 `results/case4.json`。指定相同路徑會覆寫檔案。

## GPU 排程與 log

在本目錄執行額外入口，可用兩張 GPU 平行處理不同 Case；每個實驗仍只使用一張 GPU：

```bash
python gpu_run.py
# 已安裝依賴，並將參數傳給各 Case
SKIP_INSTALL=1 python gpu_run.py -- --dtype float16 --seed 0
```

runner 使用 Python 標準函式庫，需求為 Linux、`bash` 和 `nvidia-smi`。預設使用啟動 runner 的 Python；可用 `PYTHON=/path/to/python` 指定實驗環境。依賴在排程前安裝一次，安裝輸出寫入 `setup.log`。既有 `CASE1_QUANTIZER`、`CASE2_QUANTIZER` 設定也會傳入。

每次派發前查詢 total／used／free memory，選擇 free memory 最多且嚴格大於 **12288 MiB（12 GiB）** 的 GPU。取得該卡的檔案鎖後再次檢查，並透過 GPU UUID 設定 `CUDA_VISIBLE_DEVICES`。每張卡最多一個實驗，最多同時兩個；只有一張卡時依序執行。既有 `CUDA_VISIBLE_DEVICES` 若有限制，runner 只選其中的 GPU（支援實體 index 或唯一 UUID 前綴）。

沒有符合條件的 GPU 或可取得的鎖時，每 30 秒重查，等待過程不寫入 log；可用 `--poll-seconds` 調整間隔。兩個 Part 的 runner 共用 `/tmp/edgeai-gpu-<UID>/<GPU UUID>.lock`，協調同一使用者啟動的實驗。這些鎖不會約束其他程式；12 GiB 是啟動門檻，後續仍可能因記憶體不足而失敗。

每次執行會顯示獨立目錄 `results/gpu_runs/<UTC 時間戳記>_<PID>/`，包含：

- `setup.log`：安裝輸出，或跳過安裝的紀錄。
- `runner.log`：初始 GPU 狀態、實驗開始／結束與錯誤。
- `case1.log` 至 `case4.log`：各 Case 的合併 stdout／stderr。
- `case1.json` 至 `case4.json`：原本的完整 metrics，包括儲存帳本、適用的 SVD／pruning／GPTQ 資料與量化設定，不新增位寬計算。
- `summary.json`：初始 GPU 狀態、各實驗啟動前記憶體、命令、環境設定、時間、耗時、退出碼與完整 metrics。缺少或無法解析的 metrics 會明確標記。

某個 Case 失敗時，runner 繼續其他 Case，最後以非零退出碼表示有失敗。GPU 查詢失敗會停止排程並清理執行中的實驗；Ctrl+C 或 SIGTERM 也會先終止實驗子程序再釋放鎖，並保留已取得的結果。

runner 管理 `--case` 與 `--metrics-json`，不接受從轉傳參數覆寫。原本的 shell 入口仍可直接執行；新增的 `CASE_IDS="1 3"` 與 `RESULTS_DIR=/path/to/results` 可指定 Case 和輸出位置，預設仍為四個 Case 與 `results/`。runner 自行設定這兩個變數。

## 檢查與效能量測

```bash
python -m unittest discover -s tests -v
python check_numerics.py
python exercises.py --case 3
python bench_inference.py --bits 3 --group-size 128 --output results/bench.json
```

數值檢查及完整性檢查成功時不輸出；失敗以非零狀態結束。benchmark 將效能與儲存結果寫入 `--output` 指定的 JSON。格式與計算方式見 [guide.md](guide.md)。
