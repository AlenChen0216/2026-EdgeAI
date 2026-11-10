# Part 3：AdaptQuant 數值重現練習框架

依據 `../26502_AdaptQuant_Hardware_Alig.pdf` 的 Table1、Section3.2／4.2／4.3、Algorithm1 與 AppendixB。已完成數值核心、格式選擇、reference閾值推導、權重／activation接線及評估入口。**目前尚未用正式reference模型產生閾值，也尚無Qwen重現PPL。**

2026-11-08 已核對 [HackMD 最新要求](https://hackmd.io/OAKquROlSLGr9E3-P7Nm8A?view)，原文同步於 [HW.md](../HW.md)。框架已改為 **(Weight Bits + Activation Bits) / 2 ≤5.3 bits**，兩者包含對應格式的 scale overhead；在正式評估 forward 後統計與判定，不再單獨限制權重位寬。

## 先跑不需模型的檢查

在專案根目錄啟用既有本機環境：

```bash
source ~/.venvs/edgeai-5090/bin/activate
python -m unittest discover -s lab2/part3/tests -v
python lab2/part3/mx_formats.py --bits 4
python lab2/part3/exercises.py
```

上述命令均可執行。測試包含獨立codebook oracle、MX scale邊界、NVFP4比較、threshold search暴力搜尋對照，以及小型Llama的reference推導流程；接線測試另使用固定替身。測試通過不代表正式Qwen重現成功。

## 已完成的數值與推導實作

| ID | 函式 | 輸入 → 輸出 | 參考與檢查 |
| --- | --- | --- | --- |
| P3-S1 | `tile_statistic` | tiles／mask `[N,16,32]` → statistic `[N]` | Eq.4；均勻值、單一非零、全零 |
| P3-S2 | `select_formats` | statistic、`(t1,t2)` → integer bits `[N]` | Eq.5；檢查兩個閾值的相等邊界與 CAP |
| P3-Q1 | `block_scales` | tiles／mask、format → `[N,16,1]` scale | Eq.2；每列 32 值共用，不是整個 tile 共用 |
| P3-Q2 | `round_elements` | normalized tiles／mask、format → element-grid float | Section3.2；ties-to-even、subnormal、flush、飽和 |
| P3-T1 | `derive_thresholds` | Llama-3.1-8B、`[1,2048]` token IDs → 閾值文件 | Algorithm1／AppendixB；先以小型 histogram 對照暴力搜尋 |

`fake_quantize_matrix()` 處理除以scale／還原scale、分格式dispatch、分塊暫存與shape還原。S1／S2／Q1／Q2保留輸入裝置，運算使用FP32；threshold histogram的誤差加總與prefix sums使用FP64。

T1已記錄reference operating point的bit target、成本口徑、逐層誤差與slack。Reference推導目標不能直接當成Qwen上已實現的平均位寬。`bit_rate.py`／`bit_rate_report()`負責兩個operand的格式位寬與判定；`storage.py`／`weight_storage_report()`另列權重儲存診斷，不設單獨權重門檻。

`fake_quantize_matrix()` 預設仍回傳 `(result, tile_counts)`；框架使用 `return_statistics=True` 時另回傳 int64 `[3,2]` statistics。三列依序是 FP4／FP6／FP8，兩欄依序為有效元素數、非空 32-element scale blocks 數。這只記錄既定選擇，不提供量化核心答案。

格式是 MXFP4 **E2M1**、MXFP6 **E2M3**、MXFP8 **E4M3**，不是均勻整數。`mx_formats.element_table()` 列出有限碼值，供你檢查數值規則；不負責替你決定 rounding。

## 固定閾值與 calibration-free

論文使用Llama-3.1-8B的單段2048-token WikiText-2序列，離線推導一次weight／activation各七種projection的閾值，再原封不動套用至目標模型。本實作枚舉完整兩閾值grid，保留每種projection的`[t1,t2]`。論文結果的t2皆為CAP；本次結果須依reference實測，程式不強迫刪除FP8。

`thresholds.example.json`刻意留空且`frozen=false`，不能當成正式配置。執行推導後另存`thresholds.json`，記錄corpus／split、token數、序列SHA256、程式hash及推導程序／operating point。reference revision由載入模型的設定自動記錄；無法取得時記為`unknown`，不需手動指定。`CAP`用JSON字串表示；不使用JSON Infinity。

先安裝依賴，再用正式reference模型推導一次：

```bash
cd lab2/part3
python -m pip install -r requirements.txt
python derive_thresholds.py --target-bits 5.3 --output thresholds.json
```

未指定`--token-ids`時，程式以reference模型的tokenizer讀取WikiText-2 **train**，不加special tokens，取`--token-offset`起的2048個tokens，預設offset0。模型與tokenizer使用預設版本；指定`--reference-path`時使用該本機目錄。這是明列的重現選擇，論文未公開確切片段。tokens另存為`thresholds.tokens.json`。本機已有模型可加`--reference-path /path/to/Llama-3.1-8B`；入口只載入reference model，在單張GPU執行，不使用offloading。reference權重需要Hugging Face的Llama存取權或對應本機模型。

已有2048個reference token IDs時，可改用：

```bash
python derive_thresholds.py --token-ids reference_tokens.json \
  --corpus wikitext/wikitext-2-raw-v1 --split train \
  --target-bits 5.3 --output thresholds.json
```

推導先以forward pre-hooks累積各input channel的activation二階矩，並以`sum_o W[o,c]^2`衡量activation誤差；weight誤差則以activation二階矩加權。每個tile的誤差先除以自身加權能量，再加入histogram，避免大能量tile主導全部閾值。

NVFP4比較採16-element E2M1、E4M3 block scale及FP32 tensor scale，後者固定為完整operand的`amax/(6*448)`，跨所有tile chunks共用。E4M3及E2M1皆以純PyTorch模擬nearest-even rounding，與MXFP4的32-element power-of-two scale分開處理。此兩級scale定義參照[NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/api/c/recipe.html)，不包含額外Hadamard或訓練配方。

搜尋grid為`0,0.02,...,4.50,CAP`，prefix sums枚舉25,878組有序threshold pairs；每層都保留自己的NVFP4誤差約束。先找最小bit cost，相同成本優先較多FP4，再優先較大的t2以處理空bin的等價cut。若無可行pair，回報最小worst-layer error ratio的fallback，並將`reference_met=false`。

預設reference mean operand bit target為5.3，包含E8M0 scales，排除indices／shape。activation先維持NVFP4 parity，weight依48次、0～60dB的slack二分搜尋承擔預算；若weight需低於全FP4，交換角色。離散grid下可能低於目標，不能將此reference值當作Qwen的實現位寬。`derivation.reference_target_met`及各projection的`reference_met`保存實際結果。論文未列完整數字閾值、確切片段及所有設定，因此不能宣稱是作者原始配置。

Qwen 的 weight 可依固定規則計算 statistic 與 scale；activation 可在當次 forward 動態處理。這些都不改變已固定的閾值。禁止在 Qwen 上蒐集校正資料、重新推導閾值或反覆依 test PPL 選配置。

## 執行與重現

指定環境為 Python 3.11、torch 2.5.1+cu121；額外依賴固定於 requirements。從專案根目錄進入：

```bash
cd lab2/part3
# 乾淨 T4 環境：使用已推導的固定閾值，量化、評估並統計 bit rate
bash run.sh --thresholds thresholds.json --metrics-json experiment.json
# 原始模型對照
bash run.sh --baseline --metrics-json baseline.json
```

若已在本機 5090 開發環境，使用 `SKIP_INSTALL=1 bash run.sh ...` 保留 CUDA12.8 的 torch。兩個實驗都可加 `--revision QWEN_COMMIT` 固定相同 snapshot。預設沿用老師 loader 的 float16；可用 `--dtype bfloat16` 比較論文 BF16，但必須分別保存同 dtype 的 baseline，不能為追分混用。核心運算使用 PyTorch2.5.1 已有介面，不使用 Blackwell kernel。

`evaluate.py` 的 `load_model`、`wikitext2_test_text`、`eval_ppl_wikitext2` 從 Part2 原樣複製；PPL 的 WikiText-2 test、2048 切段、loss 與尾段處理均未修改。**此處假設課程 Part3 共用 Part2 評估設定**；若助教另發入口，應改接指定版本。本資料夾可獨立執行，不會從 sibling Part2 匯入 helper。

量化範圍預設為 transformer blocks 的 q/k/v/o、gate/up/down，排除 embedding、norm 與 lm_head；論文未完整列出 eligible-layer 清單，此為明列的重現假設。權重 `[O,K]` 的 row 是 output channel，activation 的 leading batch/token 維度展平為 `[M,K]`；兩者皆沿 K 使用32元素 block，tile 為16列。非整除尺寸 padding 不參與 statistic／scale；全零 tile 的 statistic 定為0，全零／空 block 的 scale 定為1。這是邊界處理的實作約定。

這條路徑儲存浮點 fake-quantized 權重，權重只處理一次，activation 每次 forward 都處理。**不宣稱實際低位元儲存、CR 或硬體加速。** activation tile counts 是執行後的觀察，不用於重新選配置。

## Realized Effective Bit Rate：兩個 operand 平均 ≤5.3 bits

新版 HW.md 定義：

`Realized Effective Bit Rate = (Weight Bits + Activation Bits) / 2 <= 5.3`

Weight Bits 與 Activation Bits 都是**實際精度分配後、含 scale overhead 的平均有效位寬**，不是 FP16 模擬容器的大小。對完整 32-element MX block，8-bit E8M0 scale 平攤後，FP4／FP6／FP8 分別為 4.25／6.25／8.25 bits。各 operand 先加總編碼與 scale 成本再除以涵蓋的元素數，最後才將兩個平均相加除以二。即使 Weight Bits 大於 5.3，兩者平均仍可能通過；不能只由權重位寬拒絕執行。

動態 activation 在固定方法配置的指定評估 forward 中記錄格式分配、元素數與 scale 成本。入口先呼叫 `reset_activation_statistics(model)`，排除 warm-up／偵錯。這是結果量測，不是重新選閾值的 calibration；不得根據 Qwen 的 bit rate／PPL 反覆重推閾值。HW 未明定跨層與跨 batch 的平均細節，本框架採同一次指定評估中的元素加權：weight 每層計一次，activation 累計各次 forward 進入量化 Linear 的有效輸入元素。兩個 operand 各自加總後相除，不以 forward 數或層數直接平均。

位寬帳本只對有效元素計算編碼 bits，每個非空 block 都計完整 8-bit scale；部分有效的 block 仍需完整 scale，全 padding 的列不計 scale。故非整除尺寸不一定正好是名義 bits+0.25。Index、shape、padded codes、空列 scales 與容器成本不進入這個格式平均，另由儲存診斷說明。此 padding／加權方式是本框架明列的統計約定，HW 未細定，報告需保留這項口徑。論文 Table7 的 weight-only W-bits 或儲存診斷的 `effective_weight_bits` 不能直接當作兩個 operand 平均。浮點 fake quantization 也不代表已產生實際低位元 checkpoint。

### 框架統計與輸出

| 位置 | 目前行為 | 判讀方式 |
| --- | --- | --- |
| `bit_rate.py`／`bit_rate_report()` | 分別加總兩個 operand 的有效編碼與 scale bits，再計算平均 | `bit_rate.passes_limit` 才是 ≤5.3 判定；以整數比值比較，不先四捨五入 |
| `MXLinear` | 保存 weight statistics；累計非持久化的 activation statistics、元素數及 forward 次數 | activation 資料在正式評估前清零，不寫入 checkpoint |
| `run_eval.py` | 先完成指定 PPL forward，再計算 bit rate，保存兩者 | 平均超標仍保存 PPL 與帳本，再以非零狀態結束；缺任一層的 operand 統計也以非零狀態結束，不報 PASS |
| `storage.py`／`weight_storage_report()` | 另計完整 padded tiles、scales、index 與 shape 的儲存成本 | 僅診斷，無 `passes_limit` 或單獨權重預算 |

`weight_storage`的`payload_bits`已含完整tiles的padding；另有scale、每tile 5-byte index，以及每矩陣8-byte shape成本。`simulation_weight_bytes`是浮點模擬權重用量，`packed_checkpoint_written=false`表示沒有輸出打包檔。Baseline不套用bit rate判定，其`bit_rate`為null。統計測試通過不代表Qwen已實測達標。

## 報告與結果

`experiment.json` 記錄 PPL、`bit_rate`、`weight_storage`、每層 weight／activation tile 數、閾值文件 hash／來源、模型 revision 與 torch／CUDA。`bit_rate` 中的 `weight_bits`、`activation_bits`、`realized_effective_bit_rate`、`passes_limit` 是正式格式統計；`weight`／`activation` 各含逐層及全體的元素數與 payload／scale／total bits，逐層明細另列各格式元素與 scale-block 計數。缺少統計時 `bit_rate=null` 並保存 `bit_rate_error`，不可視為通過。報告必須呈現最終 PPL、兩個 operand 的位寬、平均 bit rate 及其計算。補記 GPU、Python／其他依賴、程式 hash 和完整命令。

| 項目 | 實測或設定 |
| --- | --- |
| 固定閾值來源與 hash | |
| reference model／revision／序列 hash | |
| target revision、dtype、環境 | |
| 原始模型 PPL | |
| 量化模型 PPL | |
| 與課程基準13.98差值 | |
| weight／activation 格式比例 | |
| weight／activation 各自的元素總數、統計層與評估範圍 | |
| Weight Bits、Activation Bits 各自的編碼與 scale 成本計算 | |
| (Weight Bits + Activation Bits) / 2 與 ≤5.3 判定 | |
| 另列的 index／metadata／padding 儲存成本與口徑 | |
| 格式成本帳本與浮點模擬實際 bytes 的區別 | |
| 與論文不同的假設 | |

可用圖片：小 tensor 的原始／重建值、誤差分布、實測 PPL 與基準比較。13.98 是課程與論文 Table5 的比較值，不是本次測量。用實測確認（confirm）結論，交代解釋結果的推理（logic）；不要將尚未驗證的原因寫成事實。

五項程式TODO均已完成。2026-11-09，本機Python3.11／torch2.7.0+cu128通過35項測試及`check_numerics.py`，包含CPU／CUDA rounding與小型Llama reference流程。正式Llama-3.1-8B的存取與閾值產生由使用者後續處理，因此目前未跑正式reference derivation、Qwen PPL或T4環境驗證，也未產生報告PDF。正式繳交仍需保存上述實驗結果及`314552017_report.pdf`，並在乾淨T4環境依README完整執行。
