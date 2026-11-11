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


## Appendix：產生 thresholds

現有 `thresholds.json` 可直接用於正式實驗。若需重新推導，需取得 Hugging Face 上 `meta-llama/Llama-3.1-8B` 的存取權，以及支援 bfloat16 的 CUDA GPU。安裝本專案依賴後，在 `lab2/part3` 執行：

```bash
python derive_thresholds.py --target-bits 5.3 --output thresholds_new.json
```

程式預設使用 WikiText-2 train 的 2048 個 tokens，產生 `thresholds_new.json` 與 `thresholds_new.tokens.json`。範例使用不同檔名，不會覆寫正式實驗使用的 `thresholds.json`。
