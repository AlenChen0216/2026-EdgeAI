# Part 2：壓縮與儲存計算

## 目標與評估

`common.target_linears()` 選取 decoder 的 q/k/v/o、gate/up/down projections。`record_base()` 在壓縮前記錄目標層 bytes；`storage_report()` 加總替換後的參數、buffers 及新增 tensor，計入 packed codes、scale、zero point、padding、bias 與 metadata。

`Compression Ratio = 1 - compressed bytes / original bytes`。未壓縮的目標層仍計入分子及分母。`evaluate.eval_ppl_wikitext2()` 在 WikiText-2 test 上以長度 2048 的完整區塊計算下一 token 的 PPL。

## 量化與 GPTQ

`group_weight()` 沿輸入維度分組並標記 padding。`choose_qparams()` 或 `choose_qparams_mse()` 求出每組的 scale、zero point，`quantize_values()` 產生整數 codes，`pack_codes()` 將 codes 寫入 byte stream。`PackedLinear` 在 forward 時重建當前層權重並執行 Linear；持久儲存只保留壓縮表示與必要參數。

Case 1 預設在正式入口使用 GPTQ：attention 4-bit、MLP 3-bit、group size 128。Case 2 使用 GPTQ：全部目標層 3-bit、group size 512，scale 為 FP16，zero point 以對應位寬打包。GPTQ 根據 WikiText-2 train 校正輸入建立 Hessian 近似，依序量化並補償後續權重的誤差。各 decoder 中 q/k/v、o、gate/up、down 按順序處理。

## Case 3：SVD 與因子量化

`case3_ranks()` 依全部目標權重的儲存預算選擇 rank。Whitening SVD 以校正輸入的 Gram matrix 加權原始權重，再把 `W[O,I]` 分解為 `A[O,r]` 與 `B[r,I]`。`case3_quantization_plan()` 決定各因子的量化列數；`MixedPrecisionLinear` 同時保存浮點列與 packed 列。

`SVD Compression Ratio = 1 - SVD 因子 bytes / 全部原始目標權重 bytes`，在因子量化前量測，不含 bias。最終 Compression Ratio 另依全部目標層的實際持久儲存計算，包含 bias、量化參數與 metadata。兩種數值及逐層 rank 記錄在 metrics JSON。

## Case 4：剪枝與量化

SparseGPT 對每四個連續權重選一個剪除，並補償剪枝及量化誤差。`Sparse34PackedLinear` 每四個位置保存三個量化值，以及 2-bit 剪除位置；scale、zero point 與 metadata 也計入儲存量。

`Pruned Elements` 是本次剪除的原始非零權重數除以全部原始目標權重數。`Pruning Rate` 是量化前的儲存縮減：原始精度的三個存活值、位置索引、metadata 與 bias 均計入。最終 Compression Ratio 則計算剪枝加量化後的實際 buffers；零值數量不直接視為儲存縮減。

## 相關函式

- `compress_model()` 記錄基準後依 Case 選擇壓縮流程。
- `apply_gptq()` 逐組收集輸入統計、替換 Linear 並記錄診斷。
- `apply_whitened_svd()` 依 decoder 執行 whitening、低秩分解與因子量化。
- `storage_report()` 返回總量與逐層帳本，`run_eval.py` 輸出主要結果並可保存 JSON。
