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

## 檢查與效能量測

```bash
python -m unittest discover -s tests -v
python check_numerics.py
python exercises.py --case 3
python bench_inference.py --bits 3 --group-size 128 --output results/bench.json
```

數值檢查及完整性檢查成功時不輸出；失敗以非零狀態結束。benchmark 將效能與儲存結果寫入 `--output` 指定的 JSON。格式與計算方式見 [guide.md](guide.md)。
