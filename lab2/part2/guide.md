# Part 2：Qwen3-4B 壓縮實驗引導

依據：[作業要求](../HW.md)、[run_eval.py](run_eval.py)、[evaluate.py](evaluate.py)。本文件協助你設計與檢查實驗；不提供四個 Case 的完成配置，也不預先宣稱任何方法能達標。

## 1. 先把限制寫進實驗紀錄

- 占 50%，固定模型為 **`Qwen/Qwen3-4B`**。
- 評分基礎環境是 **Python 3.11、`torch==2.5.1+cu121`**。
- 四個 Case 都從相同的原始模型開始；使用同一模型版本，避免把上一個 Case 的壓縮模型繼續壓縮當成下一個 Case。
- 壓縮率依目標 Linear layers 計算，壓縮方法新增的參數也要納入。
- 所有 PPL 都必須使用老師提供的 `run_eval.py`，不得替換或更動 PPL 評估程序。
- 禁止 KV cache 壓縮／量化、speculative decoding、parameter offloading、fine-tuning、多 GPU model parallelism，以及只降低記憶體使用、未壓縮模型的技巧。

| Case | 壓縮條件 | PPL 條件 |
| --- | --- | --- |
| 1 | Compression Ratio **> 75%** | **< 15** |
| 2 | Compression Ratio **≥ 81%** | **< 20** |
| 3 | SVD Compression Ratio **≥ 20%**；Overall Compression Ratio **= 50%** | **< 50** |
| 4 | Pruning Rate **≥ 20%**；Overall Compression Ratio **≥ 85%** | **< 50** |

注意嚴格大於、小於與大於等於的差別。Case 1 的 75% 與 PPL 15 都不算通過。

## 2. 目前檔案還缺哪些部分

開始壓縮前先讀現有程式，避免把命令列參數當成已完成的功能：

| 檔案位置 | 目前行為 | 對實驗的影響 |
| --- | --- | --- |
| `run_eval.py` 的 `main()` | 載入模型後直接評估，沒有呼叫壓縮函式 | 傳入 `--case` 並不會套用該 Case 的壓縮 |
| `--rate`、`--wbits` | 有解析參數，但目前沒有用於壓縮 | 更改旗標不會改變模型 |
| `evaluate.py` 的 `compression_rate()` | 兩條路徑目前都回傳 0 | 尚不能取得有效的壓縮率 |
| `compression_rate()` 的 docstring | 提到 `common.record_base(model)` | 目前 `part2/` 沒有這個實作，需確認是否漏發檔案 |

先向助教確認：是否有完整的 starter、預留的壓縮接入位置，以及允許實作壓縮率計算的範圍。**保留 PPL 的資料、切段與 loss 計算程序**。這些占位內容不表示你能重寫評估；也不要用輸入的目標 rate 當成量測結果。

## 3. 先跑一次原始模型

在符合版本要求、已裝好相容 dependencies 的環境中，從專案根目錄執行現有入口：

```bash
python lab2/part2/run_eval.py
```

這個命令目前評估的是原始模型；它不會替你安裝套件。先檢查現有 imports：`torch`、`transformers`、`datasets`，以及模型載入路徑需要的套件，再把實際可重現的版本寫入 `requirements.txt`。不要把未測試過的最新套件組合當成完成的環境。

紀錄模型與 tokenizer 版本、套件版本、dtype、seed、GPU、原始 PPL、執行命令與錯誤訊息。現有 loader 預設 float16、CUDA，會設為 eval mode、停用 KV cache 並關閉參數梯度。

目前提供的 PPL 計算使用 WikiText-2 raw test、`seqlen=2048`、完整長度區塊、下一 token 預測，尾端不足一段的部分不納入。這些是公平比較的條件，不能為了降低 PPL 或記憶體需求而改變。

若發生 OOM，先確認只有一個模型實例、沒有殘留前次模型或不必要的副本，再安排逐層轉換與暫存釋放。遵守 HW.md 的 offloading 與多 GPU 限制；壓縮後能否完成原定評估也是方法可行性的一部分。

## 4. 建立可信的壓縮率帳本

作業定義：

`Compression Ratio = 1 - Compressed Size / Original Size`

先列出所有目標 Linear 的名稱、weight shape、bias、原始 dtype 與 bytes，固定統計範圍。特別確認 `lm_head` 是否屬於目標、未處理的 Linear 如何計算、是否存在共享儲存；不能只選成功壓縮的層當分母。

為每層記錄下列項目：

| 項目 | 需要記錄的內容 |
| --- | --- |
| 原始儲存 | weight 與納入範圍的 bias bytes |
| 壓縮權重 | bitwidth、實際 packing、padding／對齊成本 |
| 量化資訊 | scale、zero point 的數量與 dtype |
| SVD | 每個 factor 的 shape、dtype，及額外保留的奇異值 |
| 剪枝 | 保留值、index、mask 或稀疏結構所需資訊 |
| 其他保留值 | 未壓縮 bias、例外通道、額外參數或必要 buffer |

必須區分三種量測：目標層壓縮表示的 bytes、checkpoint 檔案大小、GPU 記憶體用量。三者不相等。序列化也可能有額外開銷，報告要說清楚你採用哪種統計口徑，並與助教規範一致。

先做紙上估算，再寫方法。若原始權重是 16-bit，且暫時忽略所有額外資訊，單純 b-bit 權重的理想保留比例是 `b / 16`。這個估算僅協助排除明顯不可能的方案，不能代替實際 bytes。

例如，理想 4-bit／16-bit 只有 75% 壓縮率，加入 scale 等資訊後會更低，因此不能直接認定 4-bit 就通過 Case 1。把低位元值存成 `int8` 卻未 packing，或 fake quantization 後仍存成浮點，也不會自動得到名義位元數的儲存成本。

## 5. 先用小矩陣確認方法，再處理整個模型

把每個壓縮步驟分開驗證：

1. 確認轉換前後的 shape、device、dtype 與輸出介面。
2. 以小矩陣檢查量化／反量化或矩陣分解，確認沒有 NaN、Inf、零 scale。
3. 手算一次儲存帳本，與程式統計比較。
4. 若需要保存模型，確認重載後仍保留壓縮表示與相同 forward 行為。
5. 接上老師允許的壓縮入口，再執行完整模型與官方 PPL 評估。

forward 為了運算而反量化可以是實作的一部分，但若長期另存一份完整浮點權重，要說清楚實際保留的表示，不能在儲存統計中忽略它。

## 6. Case 1、Case 2：先從容量預算選候選配置

先問：為了達到該 Case，最多能保留原始目標層多少比例的 bytes？扣除 scales、zero points、padding 後，權重平均還剩多少 bit 預算？

接著一次只改一個主要因素，例如 bitwidth、group size 或特定類型 Linear 的精度，記錄壓縮率與 PPL。不要一開始同時改動所有因素，否則難以知道結果變化來自何處。

可做的比較：

- 固定 bitwidth，比較不同 group size 的品質與 metadata 成本。
- 固定其他設定，比較注意力與 MLP 投影層的壓縮影響。
- 若採 layer-wise 配置，檢查保留較高精度的層是否讓整體壓縮率低於門檻。

判斷原則：兩個門檻都必須通過。只讓 PPL 更低，卻超出容量預算，仍不是該 Case 的有效配置。對「哪類層較敏感」的結論，要有只改該因素的比較支持。

## 7. Case 3：分開規劃 SVD 與整體壓縮

先對單一 `[m, n]` 權重做參數數量估算。若用 rank `r` 的兩個因子表示，兩因子的參數數量是 `r(m+n)`；若另存奇異值或其他資訊也要另外計入。不是所有 rank 都能節省儲存，先推導會節省的條件。

實驗順序：

1. 先確認只有 SVD 時的壓縮率與輸出誤差，定義 SVD 比例的分母與統計時點。
2. 保留 SVD-only 的結果，確認是否達到 20% 要求。
3. 若再結合其他方法，重新統計最終 factors 與所有額外資訊的 bytes。
4. 用完整模型的官方 PPL 評估檢查品質；矩陣重建誤差小不保證 PPL 達標。

**需要向助教釐清**：HW.md 條件寫 Overall Compression Ratio **= 50%**，範例卻列 **50.4%**，未交代容許誤差；另外也沒有完整定義「SVD Compression Ratio」的統計口徑。暫時保留未四捨五入的 bytes 與比例，按文字條件設計，不自行把 `= 50%` 當成 `≥ 50%`。

## 8. Case 4：剪枝率與實際壓縮要分別證明

先定義 pruning rate 的分母、剪枝單位與涵蓋層：是元素、channel，還是 block？HW.md 未明確指定的統計方式先與助教確認。

思考以下差別：

- 將 dense tensor 中部分值設為 0，參數的儲存空間通常仍存在。
- 稀疏表示可以省下部分值，但需要 index、mask 等資訊。
- 結構化剪枝可能改變矩陣 shape，必須維持相鄰運算的維度與模型 forward 正確。

先驗證剪枝比例，再驗證壓縮表示的 bytes，最後才與其他壓縮方法組合。兩個階段的壓縮率不能直接相加；應使用最終表示重新計算。

這個 Case 至少要同時留有三項證據：pruning rate、overall compression ratio、官方 PPL。零值比例也要區分原本就有的零與這次剪除的元素，避免只靠零值總數推估剪枝率。

## 9. 讓實驗紀錄能直接支撐報告

每次執行記一列；未測量的欄位留空，不填預期結果。

| Case／實驗 ID | 目標層與方法 | bits／group size／rank／剪枝設定 | 原始 bytes | 壓縮 bytes | CR | SVD CR／剪枝率 | 官方 PPL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待測 | 待測 | 待測 | 待測 | 待測 |

報告每個 Case 都要有方法、設定、最終 CR、最終 PPL 與結果分析，再比較四個 Case 的取捨。圖表建議：

- **必要數據**：四個 Case 的最終結果表；Case 3 的 SVD 比例、Case 4 的剪枝率。
- **建議圖 1**：CR–PPL 散點圖，標示各 Case 的門檻。
- **建議圖 2**：相同設定下只改 group size、rank 或精度的一組比較。
- **有實驗才放**：分層敏感度、耗時、GPU peak memory；後兩項不能代替 CR。

每個結論都用實測資料確認（confirm），並交代方法到結果的推理（logic）。沒有做分層比較，就不要斷言特定投影層一定最敏感；沒有量測速度，就不要宣稱加速。

## 10. 包裝成一次可重現的提交

`run.sh` 必須安裝依賴、從相同原始模型分別執行四個 Case、用提供的評估程序取得 PPL，最後只在標準輸出列出各 Case 的 CR 與 PPL。安裝與除錯訊息可寫入 stderr 或日誌；不要丟掉它們，否則失敗時無法定位原因。

在全新環境驗證 `bash run.sh`，不能依賴手動先安裝套件或事先準備但未說明的壓縮模型。若需下載權重與資料，也要讓入口流程能處理並記錄來源。

提交前另有一個檔名矛盾：Part 2 章節指定 `run.sh`，Overall Submission 列 `StudentID_run.sh`。向助教確認後統一命名；釐清前可規劃共用一個實作、以另一個入口呼叫，避免兩份流程不同步。

依 HW.md 檢查 `requirements.txt`、入口腳本、`run_eval.py`、壓縮實作及 `StudentID_report.pdf` 都在繳交資料夾中。目前 `run_eval.py` 也依賴 `evaluate.py`，不要漏交。本地工作目錄為 `part2/`，最終壓縮檔範例使用 `Part2/`。
