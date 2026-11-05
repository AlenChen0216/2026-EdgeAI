# Part 2：Qwen3-4B 壓縮實驗引導

依據：[作業要求](../HW.md)、[run_eval.py](run_eval.py)、[evaluate.py](evaluate.py)。本文件協助你設計與檢查實驗；不提供四個 Case 的完成配置，也不預先宣稱任何方法能達標。

## 0. 在 5090 開發，掛載 Colab 後用 T4 驗證

主要實作放在 `.py`，實驗參數放在設定檔或命令列；notebook 用來連接 Colab server、掛載檔案系統與查看結果。日常在本機 5090 開發，每完成一個 Case 就將同一份程式複製到 Colab T4，重跑「原始模型 → 壓縮 → 指定評估」流程。

| 用途 | GPU | Python | PyTorch |
| --- | --- | --- | --- |
| 本機開發 | RTX 5090 | 3.11 | 本指南以 `2.7.0+cu128` 為起點 |
| 作業環境驗證 | Colab T4 | 3.11 | HW.md 指定的 `2.5.1+cu121` |

5090 使用 Blackwell 架構，PyTorch 從 [2.7 與 CUDA 12.8](https://pytorch.org/blog/pytorch-2-7/) 加入支援。兩邊分開建立環境；其他套件盡量鎖定相同版本，實作使用 PyTorch 2.5.1 已提供的介面。開發環境能執行，仍需由 T4 的指定環境確認相容性。

### 0.1 第一次建立本機開發環境

以下是 **5090 主機的 terminal** 命令。若尚未安裝 `uv`，依 [官方安裝方式](https://docs.astral.sh/uv/getting-started/installation/)執行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

第一次建立環境；之後開新 terminal 只需重新 `source` 啟用：

```bash
uv venv --python 3.11 --seed ~/.venvs/edgeai-5090
source ~/.venvs/edgeai-5090/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.56.2 datasets==3.6.0 accelerate==1.10.1
```

這組額外依賴是起始配置，以兩邊都能完成原始模型評估作為相容性確認。`transformers` 必須支援 Qwen3 及現有 loader 的 `dtype=` 參數；[4.56.2 文件](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/model)列有此介面。跑通後將共同依賴寫入 requirements；本機使用的 torch／CUDA 版本另外記錄，提交用的 `requirements.txt` 以驗證環境為準。

在本機檢查 Python、套件與實際 CUDA 運算：

```bash
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

應確認是 Python 3.11、`2.7.0+cu128`、RTX 5090。這裡不需要將開發程式改成 notebook；在 VS Code 選擇這個環境的 Python，再從已啟用環境的本機 terminal 執行 `.py`。

### 0.2 將 Colab server 掛載到目前 workspace

1. 在 VS Code 安裝 Google **Colab** 與 **Jupyter** 擴充套件，開啟一份獨立的連線 notebook。
2. 從 kernel 選單選 **Colab → New Colab Server → T4**，完成登入與連線。
3. 在設定啟用 **Colab: Server Mounting**，確認 **Colab: Terminal** 也已啟用。需要時執行 **Developer: Reload Window** 並重新連線。
4. 執行 **Colab: Mount Server To Workspace…**，選擇剛建立的 server。檔案總管會同時顯示本機專案與掛載的 Colab 檔案區。
5. 執行 **Colab: Open Terminal**，選擇同一個 server。在這個 terminal 執行：

```bash
pwd
nvidia-smi
mkdir -p /content/lab2
```

確認 GPU 是 T4。即使本機專案是透過 Remote SSH 開啟，也使用這個掛載與拖拉複製流程；操作說明見 [Colab 官方指南](https://github.com/googlecolab/colab-vscode/wiki/User-Guide#experimental-features)。此處掛載的是 Colab server，與 Google Drive 掛載不同。

### 0.3 用 GUI 複製程式，確認檔案版本

在掛載的 Colab 檔案區找到 `/content/lab2`，將本機 `lab2/part2`、`lab2/part3` 拖拉**複製**到這裡。第一次複製後應有 `/content/lab2/part2` 與 `/content/lab2/part3`；本機仍保留原資料夾。若已有同名資料夾，更新其中的程式與設定，確認覆寫的是本次要驗證的檔案。

- 複製 `.py`、設定檔、requirements、入口腳本與必要 helper。本機 `.venv`、`.python`、`.git` 及 CUDA 套件不需上傳；兩邊各自安裝環境。
- 每次先儲存本機檔案，再複製更新。拖拉是一次檔案複製，沒有自動雙向同步；重新命名或刪除的檔案也要同步整理，避免 Colab 載入舊模組。
- 若 terminal 建立或修改的檔案沒有出現在 GUI，重新整理掛載目錄。Colab 產生的結果與日誌也用複製方式存回本機。

在 **Colab terminal** 核對目的地：

```bash
ls -l /content/lab2/part2
sha256sum /content/lab2/part2/run_eval.py /content/lab2/part2/evaluate.py
```

在 **5090 主機 terminal** 對相同檔案執行：

```bash
cd /home/alen/2026-EdgeAI
sha256sum lab2/part2/run_eval.py lab2/part2/evaluate.py
```

兩邊對應的 hash 應相同。加入壓縮模組與設定檔後，也要核對本次實驗使用的那些檔案。執行中的程序不會因檔案覆寫而自動載入新版本；每次更新後以新的 Python 程序開始實驗。

### 0.4 在 Colab terminal 建立驗證環境

以下全部是 **Colab terminal 的 Bash 命令**，不加 notebook 的 `!` 或 `%`。若 Colab 沒有 `uv`，先在這個 terminal 執行第 0.1 節的安裝命令；本機安裝過 `uv` 不代表 Colab 也有。

```bash
uv venv --python 3.11 --seed /content/edgeai-t4
source /content/edgeai-t4/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers==4.56.2 datasets==3.6.0 accelerate==1.10.1
```

`uv` 可下載缺少的 Python 3.11，見 [環境管理文件](https://docs.astral.sh/uv/pip/environments/)。這套環境獨立於 notebook kernel；切換 notebook kernel 不會替 terminal 啟用環境。每開一個新的 Colab terminal，都重新執行 `source /content/edgeai-t4/bin/activate`。

重跑第 0.1 節的環境檢查，應看到 Python 3.11、`2.5.1+cu121`、CUDA 12.1 與 T4；CUDA tensor 運算也要成功。`torch.version.cuda` 是套件使用的 CUDA 版本，與 `nvidia-smi` 顯示的驅動支援版本可以不同。

Part 2 與 Part 3 可共用這套環境，新增依賴後要重新確認兩部分都能執行。Part 1 的 notebook 使用獨立的 PyTorch 2.6.0 環境。Colab server 被刪除或 runtime 重建後，需重新掛載、建立環境並複製程式。

### 0.5 固定每次驗證與保存結果的順序

1. 本機儲存程式，先做小矩陣、單層與完整流程的開發檢查。
2. 複製本次程式與設定到 Colab，核對路徑與版本。
3. 在 Colab terminal 啟用驗證環境，從原始模型重跑壓縮與指定評估。
4. 記錄本機程式版本／檔案 hash、模型 revision、套件版本、dtype、seed、配置、CR、PPL、耗時及量測到的峰值顯存。
5. 將結果與日誌複製回本機，確認保存後再釋放 Colab server。

5090 與 T4 的顯存容量不同，開發時避免不必要的完整模型副本，並及早在 T4 跑通完整流程。兩邊使用相同的評估 dtype，先沿用入口預設的 `float16`。如果顯存不足，檢查模型與暫存生命週期；仍遵守第 1 節對 offloading 等技巧的限制。

GPU 與套件版本不同可能造成浮點結果差異，固定 seed 也不保證逐位相同。分別保留兩邊的 baseline 與壓縮結果，報告標明環境，最終以指定環境實測確認。這些環境差異可參考 [PyTorch 數值精度說明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。

## 1. 先把限制寫進實驗紀錄

- 占 50%，固定模型為 **`Qwen/Qwen3-4B`**。
- 評分基礎環境是 **Python 3.11、`torch==2.5.1+cu121`**。
- 四個 Case 都從相同的原始模型開始；使用同一模型版本，避免把上一個 Case 的壓縮模型繼續壓縮當成下一個 Case。
- 壓縮率依目標 Linear layers 計算，壓縮方法新增的參數也要納入。
- 目標範圍只計算 transformer block 內的 Linear layers（例如 q/k/v/o、gate/up/down）；`lm_head` 不納入 Part 2 的壓縮率。
- 只壓縮部分目標層時，其他未壓縮的目標層仍以原始大小計入。
- 同一個壓縮方法可以重複用於多個 Case，但每個 Case 仍須從相同的原始模型重新執行並獨立量測。
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
| `evaluate.py` 的 `compression_rate()` | 兩條路徑目前都回傳 0 | 需依實際使用的壓縮方法自行計算壓縮率 |
| `compression_rate()` 的 docstring | 提到 `common.record_base(model)` | 目前 `part2/` 沒有這個實作，需確認是否漏發檔案 |

確認完整 starter 與預留的壓縮接入位置後，依實際使用的壓縮方法自行計算壓縮率；不要用輸入的目標 rate 當成量測結果。**保留 PPL 的資料、切段與 loss 計算程序**，不要重寫評估。

## 3. 先跑一次原始模型

先在已啟用開發環境的 **5090 主機 terminal**，從專案根目錄執行：

```bash
cd /home/alen/2026-EdgeAI
python lab2/part2/run_eval.py --dtype float16 --seed 0
```

依第 0 節掛載、複製程式並建立 T4 環境後，在 **Colab terminal** 執行相同入口：

```bash
source /content/edgeai-t4/bin/activate
cd /content/lab2/part2
python run_eval.py --dtype float16 --seed 0
```

兩個命令目前都評估原始模型，不會套用 Case 壓縮。先檢查現有 imports：`torch`、`transformers`、`datasets`，以及模型載入路徑需要的套件，再把實際可重現的版本寫入 `requirements.txt`。不要把未測試過的最新套件組合當成完成的環境。

紀錄模型與 tokenizer 版本、套件版本、dtype、seed、GPU、原始 PPL、執行命令與錯誤訊息。現有 loader 預設 float16、CUDA，會設為 eval mode、停用 KV cache 並關閉參數梯度。

目前提供的 PPL 計算使用 WikiText-2 raw test、`seqlen=2048`、完整長度區塊、下一 token 預測，尾端不足一段的部分不納入。這些是公平比較的條件，不能為了降低 PPL 或記憶體需求而改變。

若發生 OOM，先確認只有一個模型實例、沒有殘留前次模型或不必要的副本，再安排逐層轉換與暫存釋放。遵守 HW.md 的 offloading 與多 GPU 限制；壓縮後能否完成原定評估也是方法可行性的一部分。

## 4. 建立可信的壓縮率帳本

作業定義：

`Compression Ratio = 1 - Compressed Size / Original Size`

先列出 transformer block 內所有目標 Linear 的名稱、weight shape、bias、原始 dtype 與 bytes，固定統計範圍；`lm_head` 不納入 Part 2 壓縮率。未處理的目標 Linear 仍以原始大小計算，不能只選成功壓縮的層當分母。

若目標 Linear 有共享權重，依模型運作時的使用次數計算：若共享區塊的運算結果也能共用，只記一次；若共享後仍進行兩次以上運算，按實際運算次數作為倍率。

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

Overall Compression Ratio **= 50%** 是剛好 50%，不是 `≥ 50%`；允許誤差為 **±0.5 個百分點**，因此可接受約 49.5%–50.5%。保留未四捨五入的 bytes 與比例，再說明實際誤差。SVD Compression Ratio 仍須以實際保留的 SVD factors 與額外資訊計算。

## 8. Case 4：剪枝率與實際壓縮要分別證明

先定義 pruning rate 的分母、剪枝單位與涵蓋層：是元素、channel，還是 block？HW.md 未明確指定的統計方式先與助教確認。

思考以下差別：

- 將 dense tensor 中部分值設為 0，參數的儲存空間通常仍存在。
- 稀疏表示可以省下部分值，但需要 index、mask 等資訊。
- 結構化剪枝可能改變矩陣 shape，必須維持相鄰運算的維度與模型 forward 正確。

先驗證剪枝比例，再驗證壓縮表示的 bytes，最後才與其他壓縮方法組合。兩個階段的壓縮率不能直接相加；應使用最終表示重新計算。

這個 Case 至少要同時留有三項證據：pruning rate、overall compression ratio、官方 PPL。零值比例也要區分原本就有的零與這次剪除的元素，避免只靠零值總數推估剪枝率。

## 9. 讓實驗紀錄能直接支撐報告

每次執行記一列；未測量的欄位留空，不填預期結果。同一個實驗 ID 分別記錄 5090 與 T4 的結果，附上程式 hash、環境版本與執行命令；表中的正式結果取自 T4 指定環境的完整驗證。

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

在 Colab T4 的乾淨 Python 3.11 環境中，按提交說明驗證 `bash run.sh`。入口要自行安裝指定的 `torch==2.5.1+cu121` 與其他依賴，不能依賴第 0 節探索時已手動裝好的套件，或事先準備但未說明的壓縮模型。若需下載權重與資料，也要讓入口流程能處理並記錄來源。GUI 掛載與拖拉是開發時的傳檔方式，助教在取得提交資料夾後不應需要操作 VS Code 才能執行。

提交前另有一個檔名矛盾：Part 2 章節指定 `run.sh`，Overall Submission 列 `StudentID_run.sh`。向助教確認後統一命名；釐清前可規劃共用一個實作、以另一個入口呼叫，避免兩份流程不同步。

依 HW.md 檢查 `requirements.txt`、入口腳本、`run_eval.py`、壓縮實作及 `StudentID_report.pdf` 都在繳交資料夾中。目前 `run_eval.py` 也依賴 `evaluate.py`，不要漏交。本地工作目錄為 `part2/`，最終壓縮檔範例使用 `Part2/`。
