# Part 3：Calibration-Free PTQ 重現引導

主要依據：[作業要求](../HW.md)與本機[AdaptQuant論文](../26502_AdaptQuant_Hardware_Alig.pdf)。量化核心、格式選擇及固定閾值推導均已實作；執行方式與尚待產生的正式實驗結果見[README](README.md)。

2026-11-08 已核對[HackMD最新要求](https://hackmd.io/OAKquROlSLGr9E3-P7Nm8A?view)。正式條件為 **Realized Effective Bit Rate = (Weight Bits + Activation Bits) / 2 ≤5.3 bits**，兩者都計入對應格式的scale overhead。入口已接上兩個operand的統計與平均判定；程式TODO已完成，正式Qwen結果仍須固定reference閾值後評估。

## 0. 5090 開發與 Colab T4 驗證流程

本部分以 `.py` 實作數值模擬與模型轉換，先在本機 RTX 5090 檢查，再用 Colab T4 跑完整驗證。notebook 負責建立 Colab 連線與掛載 server；正式實驗從 **Colab terminal** 啟動。Part 1 則在本機編輯題目 notebook，透過 Colab kernel 執行 cells。

| 用途 | 環境 |
| --- | --- |
| 本機開發 | RTX 5090、Python 3.11；以 `torch==2.7.0+cu128` 為開發起點 |
| 指定環境驗證 | Colab T4、Python 3.11、`torch==2.5.1+cu121` |

本機需要支援 Blackwell 的套件，詳見 [PyTorch 2.7 說明](https://pytorch.org/blog/pytorch-2-7/)。環境建立與起始依賴版本沿用 [Part 2 guide 第 0 節](../part2/guide.md)；本部分增加的套件也要同時記錄與驗證。提交的 requirements 採用 T4 環境版本。

### 0.1 本機開發時先維持可移植性

已依 Part 2 guide 建立開發環境後，在 **5090 主機 terminal** 啟用：

```bash
source ~/.venvs/edgeai-5090/bin/activate
cd /home/alen/2026-EdgeAI/lab2/part3
```

先完成第 4 節的單一數值、tile、小矩陣與單層檢查。即使 5090 支援硬體低精度運算，實作仍要提供能在 T4 與 PyTorch 2.5.1 執行的數值模擬路徑；HW.md 允許 fake quantization，不要求使用 Blackwell 專屬 kernel。

固定方法配置後再取得 PPL，遵守第 2 節的 calibration-free 條件。可在另一個 reference model 上用 calibration data 推導固定規則，再原封不動套用到 Qwen3-4B；不在目標模型上重新校正。Calibration 與 fine-tuning 是不同程序，是否 fine-tuning 取決於是否訓練更新權重，不能由資料來源判定。轉到另一張 GPU 是確認相容性，不是重新根據評估 PPL 挑選格式、閾值或例外層。

### 0.2 Mount Server To Workspace 與拖拉複製

1. 在 VS Code 開啟獨立的連線 notebook，選 **Colab → New Colab Server → T4**。若已有用於 Part 2 的 T4 server，可選擇同一個。
2. 在設定啟用 **Colab: Server Mounting**，確認 **Colab: Terminal** 已啟用；需要時重新載入視窗。
3. 執行 **Colab: Mount Server To Workspace…**，將該 server 的檔案系統加入目前 workspace。
4. 執行 **Colab: Open Terminal**，選同一個 server，執行 `nvidia-smi` 確認是 T4，並用 `mkdir -p /content/lab2` 建立目的地。
5. 在 GUI 找到掛載的 `/content/lab2`，將本機 `lab2/part3` 拖拉**複製**到這裡，保留本機原始檔。若助教確認共用 Part 2 評估，也複製所需的 `part2` 評估檔與 helper，維持相對目錄。

操作依據 [Colab 官方指南](https://github.com/googlecolab/colab-vscode/wiki/User-Guide#experimental-features)。此處掛載 Colab server，與 Google Drive 掛載不同；使用 GUI 複製檔案，也適用於目前透過 Remote SSH 操作 5090 專案的流程。

只複製程式、設定、requirements 與必要 helper；Python 環境、CUDA 套件在 Colab 另外建立。拖拉沒有自動同步效果：本機修改後需儲存並重新複製，重新命名或刪除也要在 Colab 更新。用檔案 hash 核對兩邊的實作與設定，避免評估到舊版本；具體命令見 Part 2 guide 第 0.3 節。

### 0.3 在 Colab terminal 驗證與取回結果

先依 Part 2 guide 第 0.4 節建立 `/content/edgeai-t4`。如果兩部分共用的 T4 環境已建立，每次開新的 **Colab terminal** 只需重新啟用：

```bash
source /content/edgeai-t4/bin/activate
cd /content/lab2/part3
python - <<'PY'
import sys
import torch

print(sys.executable)
print(sys.version)
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.ones(1, device="cuda") + 1)
PY
```

應確認 Python 3.11、`2.5.1+cu121`、T4 與成功的 CUDA 運算。這些是 terminal 的 Bash／Python 命令，不加 notebook 的 `!`；notebook kernel 與 terminal 的 Python 環境分開選擇。

目前 `part3/` 已有完整量化實作與獨立評估入口，暫以 Part2 原樣複製的 PPL 函式評估。依 README 產生固定reference閾值後，從這個 terminal 執行與本機相同的參數，重跑「相同原始模型 → 固定方法配置 → 指定 PPL 評估及 bit rate 統計」。入口在評估前清零 activation 記錄，評估後才計算平均並判定。若助教另發 Part3 入口，改接指定版本。每次更新程式後以新程序執行。

兩邊使用相同 dtype，若共用現有 Part 2 入口則先沿用預設 `float16`；也固定模型 revision、tokenizer、seed 與方法配置。保存兩邊的原始與量化 PPL、程式 hash、套件版本及量測到的耗時、峰值顯存。不同硬體與套件可能有數值差異，最終報告以 T4 指定環境的實測結果為主。

在掛載的 Colab 檔案區將日誌、結果與圖片複製回本機，再釋放 server。掛載不會讓 `/content` 永久保存；runtime 重建後需重新建立環境與上傳程式。

## 1. 先確認你要重現什麼

- Part 3 占 20%：重現結果 15%、報告 5%。
- 固定模型為 **`Qwen/Qwen3-4B`**。
- 基礎環境為 **Python 3.11、`torch==2.5.1+cu121`**。
- 目標是 **target-model calibration-free post-training quantization（PTQ）**。
- 可重現 AdaptQuant，也可選符合要求的其他方法。
- 使用 T4 時不要求硬體專用低精度 kernel；允許 fake quantization 模擬低精度數值並評估品質。
- 最新上限：**Realized Effective Bit Rate = (Weight Bits + Activation Bits) / 2 ≤5.3 bits**。兩個 operand 各自使用精度分配後的平均有效位寬，都包含 scale overhead。

不能把新規則解讀為 Weight Bits 與 Activation Bits 各自都必須 ≤5.3。例如 Weight Bits=6.25、Activation Bits=4.25，平均是 5.25；若其他條件符合，就滿足這項上限。先各自加總格式編碼與 scale 成本、除以各自涵蓋的元素數，再將兩個平均相加除以二；不要把 weight 與 activation 元素混在同一分母，也不要先平均各層位寬。

對完整 MX block，8-bit E8M0 scale 由 32 個元素分攤，因此 FP4／FP6／FP8 的有效位寬分別為 4.25／6.25／8.25。這適用於 weight 與 activation。新版 HW 明列 scale overhead；索引、shape、packing 與 padding 的額外儲存成本應另列並說明口徑，不能直接把舊版含索引的權重儲存位寬當成新的兩個 operand 平均。

`bit_rate.py`／`bit_rate_report()` 根據每層的有效元素與非空 block 計數計算格式位寬。每個非空 block 計完整 8-bit scale，含部分有效的 block；全 padding 的列不計 scale，padded codes 不計入格式 payload。Index、shape 與完整 padded 表示另列於 `weight_storage`，不進入平均判定。`run_eval.py` 在評估後輸出兩個 operand 與平均 bit rate；超標時保存指定的 metrics JSON 與 PPL，再以非零狀態結束。缺任一層的 operand 統計時也不報 PASS。介面及成本口徑見 [README](README.md)。

先取得論文全文或作者實作，再決定方法。不要只看到 PPL 門檻就任意提高精度或縮小量化範圍；重現實驗必須有清楚的方法、數值格式與對照設定。

HW.md 沒有完整列出 weight／activation bitwidth、量化層範圍與各格式細節。若改用其他方法，先確認助教接受的設定與比較規則，避免把不同壓縮強度的結果當成公平重現。

## 2. 將 calibration-free 轉成可檢查的條件

對每個設定寫下「它從哪裡來」：

| 設定 | 需要回答的問題 |
| --- | --- |
| scale、zero point | 依哪條固定規則產生？需要資料集或先跑目標模型嗎？ |
| clipping threshold | 來自論文固定設定，還是在 Qwen3-4B 上搜尋？ |
| precision／format allocation | 是否依目標模型的 activation 或品質結果挑選？ |
| group／tile size | 是否是原方法固定規格？ |
| 例外層 | 是否依原方法處理，還是看到 PPL 後才決定？ |

Calibration-free 的重點是目標模型不參與 calibration configuration 的推導。本框架依論文在 `meta-llama/Llama-3.1-8B` 上用一段 2048-token 文字推導一次閾值，再原封不動套用到 `Qwen/Qwen3-4B`；若選 WikiText-2 train split，需記錄這項片段選擇，而非宣稱已取得作者使用的確切片段。目標模型不接觸任何 calibration data。這是離線 calibration，沒有訓練更新權重。

同樣地，反覆查看評估 PPL 再挑閾值、格式配置或例外層，會讓評估資料參與設定選擇。應先依方法固定配置，再跑正式評估；偵錯時可以找實作錯誤，但要區分修正錯誤與依 PPL 調整配置。

在固定配置的評估 forward 中記錄動態 activation 的格式分配，用來計算 bit rate，是結果量測；不能把量測結果回饋成 Qwen 專屬閾值搜尋。正式評估前重設 activation 計數，避免混入 warm-up、偵錯或其他資料的 forward。HW 未明定 activation 跨層／跨 batch 的平均細節，應在報告明列統計方式；建議對同一次指定評估中的元素加權，而非把每次 forward 當成相同大小。

讀取權重是量化實作的必要部分，但「只讀權重」本身不足以證明所有搜尋都符合規定。若使用其他模型或資料作 calibration，應固定來源模型、資料、token 長度、推導規則與輸出配置，並在報告中確認 target model 沒有參與推導，也沒有用 PPL 做後續調參。

## 3. 讀論文時先完成一張方法表

先定位方法、演算法、量化格式、實驗設定及附錄，不急著直接寫完整模型。

| 欄位 | 從論文／作者實作填入的內容 |
| --- | --- |
| 量化對象 | 權重、activation，或兩者；涵蓋與排除哪些層 |
| 格式 | bitwidth、可表示值、指數／尾數配置 |
| 粒度 | per-tensor、per-channel、group、tile／block 的維度與大小 |
| 尺度 | 尺度表示法、計算規則、是否共享 |
| 數值規則 | rounding、飽和、最大最小值、零值與邊界處理 |
| 選擇規則 | 格式或精度如何決定，依據是否來自目標模型校正 |
| 推論 | 反量化位置、運算與累加 dtype |
| 品質比較 | 模型版本、tokenizer、資料集、sequence length、PPL 算法 |
| 重現來源 | 論文章節／表格、程式版本、固定設定檔 |

不要假設所有 4-bit 格式都是均勻整數量化，也不要只由方法名稱推測 tile 或 scale 規格。Part 1 的線性量化是概念基礎；能否直接重用，必須由目標格式判斷。

若選其他方法，額外整理三點：方法來源、為什麼不需要目標模型校正、與 AdaptQuant 的主要差異。報告需要簡短介紹這些內容；若重現 AdaptQuant，HW.md 明確表示不必再次介紹方法。

## 4. 先驗證數值模擬，再替換模型層

fake quantization 的概念是把數值映射到目標格式，再以浮點表示映射後的值。此處的目的在於模擬誤差；不能直接由浮點 tensor 的大小宣稱實際低位元儲存或硬體加速。

建議依序驗證：

1. **單一數值**：零、正負值、格式端點、超出範圍與 rounding 邊界。
2. **小群組／tile**：共享尺度是否只使用指定範圍內的值；全零群組是否正常。
3. **非整除維度**：padding 是否影響尺度，還原後是否移除 padding。
4. **小矩陣**：shape、dtype、device 正確，沒有 NaN／Inf。
5. **單一層**：輸入輸出介面不變，量化範圍與排除層符合方法。
6. **完整模型**：確認所有指定層都已處理，沒有漏掉或重複轉換。

若方法同時量化 activation，必須確認指定位置的 activation 確實經過模擬；只處理權重不能當成完整重現。方法若使用推論當下的動態尺度，要區分這與事先蒐集資料選參數的校正程序，依方法及課程規定說明。

不要從 Part 1 的 STE 題目推論這裡需要訓練。PTQ 的品質重現不需要為了 fake quantization 更新權重；若加入訓練，也不能再直接稱為原本的 PTQ 重現。

## 5. 固定配置後才取得正式 PPL

依第 0 節，先在 5090 與 Colab T4 各自用相同模型與評估設定跑原始模型，分開記錄 baseline；再從相同原始權重套用固定的方法配置。每次正式驗證都在 T4 指定環境重跑量化與評估，將程式 hash、模型 revision、tokenizer、dtype、seed、所有量化設定與完整命令保存下來。

HW.md 說 Part 3 使用統一的 Qwen3-4B 與 PPL 設定。目前 `part3/run_eval.py` 使用從 Part2 原樣複製的 loader、資料與 PPL 函式；這個共用設定是假設，仍需核對是否另有指定版本，不能自行換資料集或切段方式。

兩部分的入口均已接上實作。Part2 不帶 case、Part3 明確指定 `--baseline` 才是原始模型評估。Part3 的 JSON 以 `bit_rate.passes_limit` 判定兩個 operand 平均，權重儲存診斷不設門檻。正式結果必須確認 weight 與 activation 都依方法處理，並維持指定評估程序。

作業列出的 **AdaptQuant PPL 13.98 是課程比較基準**，不是你已完成的結果。論文原表格是否使用完全相同設定，仍需讀取原文確認。

| 實測 PPL | 重現表現得分 |
| --- | ---: |
| ≤ 15 | 15 |
| 15 < PPL ≤ 16 | 12 |
| 16 < PPL ≤ 17 | 9 |
| 17 < PPL ≤ 18 | 6 |
| 18 < PPL ≤ 19 | 3 |
| > 19 | 0 |

這張表對應 15% 的表現部分，報告另占 5%。Part 3 與 Part 2 的 PPL 邊界不同：此處 PPL 等於 15 屬於最高一級。

## 6. 結果不佳時如何定位問題

先檢查實作與實驗條件，再解讀品質差異：

| 現象 | 優先檢查 |
| --- | --- |
| PPL 幾乎沒有變化 | 指定層是否真的量化、是否誤載原始權重、精度設定是否符合方法 |
| PPL 明顯惡化 | scale 的軸、格式可表示值、rounding／clamp、是否重複轉換 |
| NaN 或 Inf | 全零群組、除零、指數範圍、中間運算 dtype |
| 只有部分尺寸出錯 | tile 邊界、padding、reshape 與還原順序 |
| 與他人結果不同 | 模型 revision、tokenizer、量化設定、評估資料與切段是否一致 |
| T4 上很慢 | 是否反覆轉換整個權重、是否建立大量暫存、是否用了逐元素迴圈 |

若為了診斷做額外實驗，要在紀錄中標明用途與是否符合正式配置。不能依評估分數反覆選擇配置後，仍宣稱設定完全不依賴目標模型校正。

## 7. 報告要留下哪些數據與圖片

必要內容：方法說明（依 HW.md 對所選方法的要求）、最終 PPL、**最終 realized effective bit rate 及其計算方式**、觀察與討論。建議以這張結果表作為核心：

| 項目 | 設定／結果 |
| --- | --- |
| 模型與 revision | 待填 |
| GPU、Python、PyTorch 與 CUDA 套件版本 | 待填；區分 5090 開發與 T4 驗證 |
| 程式／配置 hash 與執行命令 | 待填 |
| 方法與來源版本 | 待填 |
| weight／activation 格式及量化範圍 | 待填 |
| group／tile、scale 與 rounding 規則 | 待填 |
| 配置來源與 calibration-free 說明 | 待填 |
| 原始模型 PPL | 待測 |
| 量化模型 PPL | 待測 |
| Weight Bits（含 scale；元素數與加總成本） | 待測 |
| Activation Bits（含 scale；評估範圍、元素數與加總成本） | 待測 |
| (Weight Bits + Activation Bits) / 2 與 ≤5.3 判定 | 待測 |
| index／shape／padding 等另列成本與統計口徑 | 待填 |
| 與課程基準 13.98 的差值 | 待測 |

可補充的圖表與其用途：

- 小 tensor 的原始值／fake-quantized 值對照：檢查數值格式是否照規格運作。
- 量化誤差分布或飽和值比例：支持你對誤差來源的觀察。
- 實測 PPL 與課程基準的比較：呈現最終重現結果。
- 如果有做實作驗證，列出修正前後的錯誤現象與核對結果；不要把偵錯誤差當成新方法的品質收益。

圖片不是 HW.md 明列的強制項目，選擇能支持敘述的即可。報告應描述你實際遇到的困難、觀察到的品質變化、可能原因與限制。結論要用數據確認（confirm），原因分析要有推論依據（logic）；無法證實的原因應明確標為推測。

## 8. README 與最終重現檢查

繳交目錄至少包含：

```text
Part3/
├── README.md
├── requirements.txt
├── StudentID_report.pdf
└── 實作與必要設定檔
```

`README.md` 是重要提交項目，要讓助教依序完成：重建環境、安裝依賴、取得模型、執行量化、使用指定程序評估、重現報告中的 PPL 與 realized effective bit rate 計算。若提供 `run.sh`，也要交代執行目錄、參數、下載與輸出位置。

將提交資料夾複製到 Colab T4，在乾淨的 Python 3.11 環境照 README 完整跑一次，確認會安裝 `torch==2.5.1+cu121` 與所需依賴，無須未記錄的手動操作。若引用其他資料夾的評估檔或 helper，提交時必須一起包含，不能依賴你本機完整專案樹才可執行。掛載與拖拉是開發時的傳檔方式；助教拿到資料夾後應能依 README 在 terminal 執行完整流程。

最終區分清楚：哪些是原方法規則、哪些是 T4 的數值模擬實作、哪些差異可能影響重現結果。本地 `part3/` 是工作目錄，繳交資料夾依 HW.md 使用 `Part3/`。
