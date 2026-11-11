# Part 3：格式選擇與位元率

## 數值格式

`tiles.to_tiles()` 將矩陣切成 16×32 tiles，並以 mask 標記有效元素；`tiles.from_tiles()` 還原原始形狀。`tile_statistic()` 依有效值計算 max/RMS statistic，`select_formats()` 依 frozen thresholds 選擇 MXFP4 E2M1、MXFP6 E2M3 或 MXFP8 E4M3。全零 tile 的 statistic 為 0。

`block_scales()` 為每列最多 32 個值計算 E8M0 power-of-two scale，空列與全零列的 scale 為 1。`round_elements()` 將正規化後的值以 nearest-even 規則映射至格式可表示值，處理 subnormal、飽和與 padding。`fake_quantize_matrix()` 分塊處理 tiles，回傳浮點重建值與格式計數。

## 固定 thresholds

`derive_thresholds()` 在 reference 模型的單段 token 序列上收集 activation 二階矩及加權量化誤差，建立 histograms；`candidate_table()` 與 `solve_operand()` 在固定 grid 搜尋各 projection 的閾值。推導結果包含 weight 和 activation 的 q/k/v/o、gate/up/down 各一對 thresholds，並記錄模型、資料與序列來源。

`load_thresholds()` 檢查 frozen 狀態、來源與所有 threshold pairs；`CAP` 表示無限大。Qwen 的權重與 activation 使用這些固定閾值計算當下格式選擇，不依 Qwen PPL 或其資料重新搜尋 thresholds。

## 模型與評估

`quantize_model()` 將目標 Linear 替換為 `MXLinear`。權重建立時執行 fake quantization，activation 在每次 forward 時量化；運算及模擬權重使用浮點 tensor，不代表實際低位元 checkpoint 或硬體加速。

`run_eval.py` 載入模型及 thresholds，先清除 activation 統計，執行 WikiText-2 test PPL 評估，再統計位元率。`--baseline` 保留原始模型且不計算量化位元率。詳細來源、設定、格式數量與逐層帳本可由 `--metrics-json` 保存。

## Realized Effective Bit Rate

`bit_rate_report()` 分別加總 weight 與 activation 的有效元素編碼位數及 scale 位數，再各除以其有效元素數：

```text
Weight Bits     = (weight payload bits + weight scale bits) / weight elements
Activation Bits = (activation payload bits + activation scale bits) / activation elements
Realized Effective Bit Rate = (Weight Bits + Activation Bits) / 2
```

每個非空 32-element block 計入 8-bit scale；部分有效的 block 仍計完整 scale。跨層與 activation forward 依有效元素數加權。完整 block 的 MXFP4／6／8 位元率分別為 4.25／6.25／8.25 bits；非整除尺寸的結果取決於實際有效元素數。

`storage.weight_storage_report()` 另統計 padded tile、scale、index 與 shape 的儲存成本，與上述格式位元率採不同口徑。位元預算以 realized effective bit rate 是否不超過 5.3 bits 判定；缺少任一 operand 的統計時無法判定。
