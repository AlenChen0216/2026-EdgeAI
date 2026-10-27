# Report 補寫建議與截圖清單（Part 1～4）

本文件依 [題目](./HW.md)、[目前 report](./report.md)、目前程式碼與既有量測整理。下方「建議文稿」可貼入對應題目；「截圖指引」「修訂提醒」是寫作備忘，不必全部放進 PDF。建議圖片名稱尚未代表圖片已存在，請完成截圖後再加入 report。

## 資料來源與量測口徑

| 資料 | 本文件如何使用 |
| --- | --- |
| `pictures/naive_benchmark.png`、`cublas_benchmark.png`、`coalesced_benchmark.png` | Part 1、2 原有的 1024³／2048³／4096³ benchmark；保留這一組數值，不混用本次 `all` 的結果 |
| `reports/ncu_<kernel>_2048_2048_2048.ncu-repz` | Part 1～3 既有、各一次 kernel launch 的 profiler 數據；本次僅離線匯入 |
| [本次編譯紀錄](./results/adv_build_20261021.txt) | 目前 source 的 SHA-256、編譯選項、compiler 版本、registers、static shared memory、spill |
| [Part 3 本次 benchmark](./results/adv_part3_benchmark_20261021.txt) | 2026-10-21（UTC+08:00），2048³，`all`，20 次計時 launch 的中位數 |
| [Part 4 本次 benchmark](./results/adv_part4_benchmark_20261021.txt) | 同日，4096³，目前 chain 各測三輪，每輪 20 次 launch，取三輪中最佳的中位數 |
| [既有完整 grade](./results/part4_warptiling_grade.txt) | 舊的 correctness、sanitizer 與 10/10 評分紀錄；與本次單純 benchmark 分開呈現 |

本次 benchmark 使用目前 source 重新編譯的暫存執行檔 `/tmp/edgeai-adv-20261021-b_vqm4jt/sgemm`，未覆寫原有 `sgemm`。編譯選項與 Makefile 相同：`-O3 -std=c++17 -arch=native -use_fast_math -lineinfo -Xptxas -v`，CUDA 13.4.92、目標 `sm_120`。各 kernel 的 registers 與 static shared memory 和原有編譯紀錄一致。

新 benchmark 均為 RTX 5090、alpha=1、beta=0、warmup=3、reps=20，保留 harness 預設的每次計時前 L2 flush。程式以排序後的 `times[reps/2]` 回報中位數；20 次時為第 11 筆。Part 4 九次 benchmark 均保留 correctness check。本次未執行新的 profiling，也未重跑 grade 或 sanitizer。

既有 profiler 檔案註明未固定 GPU 頻率，因此數據是該次量測的觀察。Part 3 表格會明確分開本次 benchmark 時間與舊 profiler 時間；不能把兩者當成同一輪量測，或將跨輪差異全歸因於程式變更。

### 截圖共通操作

1. 在主機的 Nsight Compute GUI 使用 **File → Open File** 開啟既有 `.ncu-repz`。本環境 Nsight Compute 2026.3.1 可直接讀取，無須改名或解壓；主機端建議使用相容版本。此格式是 zstd 壓縮的 profiler report。[NVIDIA GUI 文件](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#main-menu)
2. 選擇對應的 `sgemm_*` launch，在 **Details** 展開指定 section。截圖保留檔名或 kernel/template 參數、2048³ 的識別資訊、指標名稱、單位與絕對值；關閉 baseline 差值顯示，避免把百分比變化當成原值。
3. 若圖表只顯示縮寫或數值被四捨五入，可在 **Raw** 頁搜尋下文提供的完整 metric 名稱，截取原值；Raw 可篩選指標並顯示單位。[NVIDIA Raw 頁文件](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#raw-page)
4. 圖片放在 `pictures/`，每張圖的 caption 寫明 kernel、M/N/K、畫面 section 與用於回答哪一題。不同 section 可合併在同張清晰截圖，不必為了張數而拆圖。

本文件只提供開啟既有報告、顯示既有紀錄及 benchmark 的操作；不提供新的 profiling 執行步驟。

## Part 1 — Naive kernel

### Q1：benchmark 截圖與建議文稿

**可沿用**：`pictures/naive_benchmark.png`、`pictures/cublas_benchmark.png`。兩張皆包含三種尺寸、時間、GFLOP/s 與 PASS，足以回答此題。排入 PDF 時確保最後一列未被裁掉。

建議圖說：

- 「圖 1：naive 在 1024³、2048³、4096³ 的 benchmark 與 correctness 結果。」
- 「圖 2：同三種尺寸的 cuBLAS FP32 benchmark，作為效能參考。」

**建議文稿：**

在 RTX 5090 上，以相同的 harness 預設設定比較 naive 與 cuBLAS。每次計時前清除 L2，回報 20 次 launch 的中位數；三種尺寸的 correctness 均為 PASS。結果整理如下，原始終端輸出如圖 1、圖 2。

| M=N=K | naive 時間（ms） | naive GFLOP/s | cuBLAS 時間（ms） | cuBLAS GFLOP/s | naive／規格表 FP32 峰值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2.9665 | 723.9 | 0.0594 | 36,157.8 | 0.691% |
| 2048 | 21.1149 | 813.6 | 0.2888 | 59,493.7 | 0.776% |
| 4096 | 141.6346 | 970.4 | 1.9302 | 71,203.0 | 0.926% |

### Q2：naive 的瓶頸與峰值利用率

**需補截圖：**

開啟 `reports/ncu_naive_2048_2048_2048.ncu-repz`，選 `sgemm_naive`。

| 建議檔名 | GUI 位置 | 必須顯示的數值 | 建議圖說 |
| --- | --- | --- | --- |
| `part1_naive_sol.png` | Details → GPU Speed Of Light Throughput | Compute 11.61%、Memory 95.79%、L1/TEX 99.38%、L2 2.96%、DRAM 0.14% | 「naive 2048³：Memory Throughput 接近峰值，FP32 運算並未飽和。」 |
| `part1_naive_memory.png` | Memory Workload Analysis → L1/TEX 表格的 **Global Loads** 列；另保留整體 hit rate 與 Device Memory throughput，必要時拆圖或補 Raw | Requests 537,001,984、Sectors 8,862,564,352、Sectors/Req 約 16.50；整體 L1 hit 99.23%、L2 hit 97.81%；DRAM 實際吞吐約 2.39 GB/s | 「naive 2048³：global-load sectors/request、cache hit rate 與實際 DRAM 吞吐量。」 |
| `part1_naive_warp.png` | Warp State Statistics → **Warp State (All Cycles)** | Stall LG Throttle 201.54、Warp Cycles Per Issued Instruction 225.69；約占 89.3% | 「naive 2048³：主要等待原因為 LG Throttle。」 |

**建議文稿：**

在 2048³ 的 Nsight Compute 報告中，naive 的 Memory Throughput 為 **95.79%**，比 Compute (SM) Throughput 的 **11.61%** 更接近峰值。Speed Of Light 顯示 L1/TEX Cache Throughput 為 **99.38%**、L2 為 **2.96%**、DRAM 為 **0.14%**；Warp State Statistics 則以 **Stall LG Throttle** 為主，每 issued instruction 平均等待 **201.54 cycles**，占總平均 **225.69 cycles** 的約 **89.3%**。LG Throttle 表示 local/global memory 操作的指令佇列壓力，支持此 kernel 主要受記憶體存取處理限制的判斷。

naive 將相鄰 lanes 對應到不同的 row。在固定 k 時，A 的位址相差 K 個 floats，B 則是同一個位址。其 global-load Sectors/Req 為 **16.50**，顯示 warp 的 A 存取需要大量 sectors。即使整體 L1 hit rate 達 **99.23%**，每次指令仍需處理這些存取與 sectors；高 cache hit rate 並不能消除存取指令和交易處理的成本。

因此，這裡的 memory 限制主要在 L1／load-store 存取路徑，並不表示 DRAM 頻寬已用滿。Profiler 顯示實際 DRAM 流量約 **2.39 GB/s**，遠低於規格表的 1,792 GB/s。naive 的瓶頸也不是 FP32 math 飽和：2048³ benchmark 僅達 **813.6 GFLOP/s**，以報告採用的 **104,800 GFLOP/s** 為分母，得到

$$
\frac{813.6}{104800}\times100\%\approx0.776\%.
$$

同樣計算，1024³ 與 4096³ 分別約為 0.691% 與 0.926%。截圖內的 `%peakF` 使用 harness 由裝置時脈估算的 112,282 GFLOP/s，故與上述規格表百分比不同。Compute (SM) Throughput 也涵蓋多種 SM 管線，不能直接當成 FP32 GFLOP/s／規格峰值。

**修訂提醒：** benchmark 的 `GB/s` 是「理想最低資料量 ÷ 時間」，不是實測 DRAM traffic。此處實際 DRAM GB/s 的來源是 profiler 的 `dram__bytes.sum.per_second`；即使兩者數值接近，也應區分定義。

## Part 2 — Coalescing

### Q1：benchmark 截圖與建議文稿

**可沿用**：已有 `pictures/coalesced_benchmark.png`，目前尚未加入 report 的 Part 2 Q1，直接補入即可。

建議圖說：「圖 3：coalesced 在 1024³、2048³、4096³ 的 benchmark，三種尺寸皆通過 correctness。」

**建議文稿：**

只改變 thread 到輸出座標的對應後，coalesced 的 benchmark 如圖 3。與 Part 1 原有截圖中的 naive 結果比較如下；加速比為 naive 時間除以 coalesced 時間。

| M=N=K | naive（ms） | coalesced（ms） | coalesced GFLOP/s | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 2.9665 | 0.4116 | 5,216.8 | 7.207× |
| 2048 | 21.1149 | 2.9563 | 5,811.2 | 7.142× |
| 4096 | 141.6346 | 21.0534 | 6,528.1 | 6.727× |

### Q2：用 sectors/request 解釋加速

**需補截圖：**

開啟 `reports/ncu_coalesced_2048_2048_2048.ncu-repz`，選 `sgemm_coalesced`。

| 建議檔名 | GUI 位置 | 必須顯示的數值 | 建議圖說 |
| --- | --- | --- | --- |
| `part2_coalesced_loads.png` | Memory Workload Analysis → L1/TEX → **Global Loads** | Requests 537,001,984、Sectors 1,342,701,568、Sectors/Req 約 2.50 | 「coalesced 2048³：global-load sectors/request 由 16.50 降至 2.50。」 |
| `part2_coalesced_warp.png` | Warp State Statistics → Warp State (All Cycles) | LG Throttle 18.28、總 cycles/instruction 32.80；約占 55.7% | 「coalescing 後 LG Throttle 的平均等待降低。」 |

第一張和 Part 1 的 `part1_naive_memory.png` 配對使用，不需重截 naive。

**建議文稿：**

兩版的計算量、每個輸出元素的 k 迴圈與邏輯載入量相同，但 warp 內同時發出的位址改變。naive 中 `threadIdx.x` 對應 row，因此同一 warp 對 A 的存取相差 K 個 floats、對 B 存取相同元素；coalesced 則將 `threadIdx.x` 對應 col，使 warp 對 A 讀取相同元素、對 B 讀取連續 32 個 FP32 元素。

2048³ 的 profiler 結果如下，均取 L1/TEX 表格的 **Global Loads** 列：

| 指標 | naive | coalesced |
| --- | ---: | ---: |
| Global-load requests | 537,001,984 | 537,001,984 |
| Global-load sectors | 8,862,564,352 | 1,342,701,568 |
| sectors/request | 16.5038 | 2.5004 |
| Stall LG Throttle（cycles／issued instruction） | 201.54 | 18.28 |

requests 數相同，sectors/request 卻由約 **16.50 降至 2.50**，表示相同載入指令需求可以用較少 sectors 服務。只看每個 k 的兩筆來源讀取，naive 約需 A 的 32 sectors 加 B 的 1 sector，平均為 `(32+1)/2=16.5`；coalesced 則約需 A 的 1 sector 加 B 的 4 sectors，平均為 `(1+4)/2=2.5`。量測包含其他 global loads，例如 C 的讀取，因此不必與這個主迴圈估算完全相等。

平均 2.5 小於連續 32 個不同 FP32 元素通常需要的 4 sectors/request，是因為 A 的同址讀取只需一個 sector；不能把 A、B 合併後的平均當作「每次 B 載入只用了 2.5 sectors」。Sectors/Req 的意義與同址讀取的解讀可參考 [NVIDIA L1/TEX 表格說明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#l1-tex-cache)。

LG Throttle 從 201.54 降至 18.28 cycles／issued instruction，也與存取處理壓力下降一致。這些數據支持 coalescing 造成加速；sectors 減少比例不必與總時間加速比完全相同，因為計算、指令發出、C 寫回與其他等待仍有成本。

### Q3：哪一層最忙，重複讀取由哪裡服務

**需補截圖：**

| 建議檔名 | GUI 位置 | 必須顯示的數值 | 建議圖說 |
| --- | --- | --- | --- |
| `part2_coalesced_memory.png` | Memory Workload Analysis → Chart／Memory Tables 的各層 throughput 與整體 cache hit rate；必要時用 Raw 核對 elapsed 指標 | L1/TEX **82.41%**、L2 **12.11%**、DRAM **0.96%**；整體 L1 hit **94.96%**、L2 hit **97.74%** | 「coalesced 2048³：L1/TEX 最忙，多數重複載入由快取服務。」 |
| `part2_coalesced_compute.png` | Compute Workload Analysis，包含下方分析訊息或 LSU 管線圖 | LSU 約 **82.4%**；若同時引用 Speed Of Light，Compute (SM) 為 **82.41%** | 「SM throughput 的高值主要來自 LSU，不代表 FP32 管線達 82%。」 |

**建議文稿：**

coalesced 的 Memory Workload Analysis 顯示 L1/TEX throughput 為 **82.41%**、L2 為 **12.11%**、DRAM 為 **0.96%**，因此最忙的是 **L1/TEX**。整體 L1 hit rate 為 **94.96%**，L2 hit rate 為 **97.74%**。這些 hit rate 分別以各層接收到的 sector 請求為基準，不能直接相加。

雖然程式在邏輯上讓每個 A 元素被讀取 N 次、每個 B 元素被讀取 M 次，這些 global loads 並不都需要到 DRAM。warp 內對 A 的同址讀取可合併，跨指令的重複資料大多命中 L1；L1 miss 後，很多請求又能由 L2 服務。2048³ 時每個 FP32 矩陣為 16 MiB，A、B 合計 32 MiB，小於裝置回報的 96 MiB L2，具備容量上可重用的條件，但不代表一定完全常駐。

naive 的 L1 hit rate 更高（99.23%），效能卻更低，說明 hit rate 不是唯一指標：coalesced 讓相同 requests 的 sectors 數大幅減少，因此即使 hit rate 稍降，存取處理仍更有效率。另外，Compute (SM) Throughput 的 82.41% 不能解讀成 FP32 math 接近峰值；Compute Workload Analysis 指出高利用率主要來自 **LSU**，與頻繁的 load/store 操作一致。

**數值口徑提醒：** Speed Of Light 的 `L1/TEX Cache Throughput` 是 **83.94%**，使用 `l1tex__throughput.avg.pct_of_peak_sustained_active`；Memory Workload Analysis 的 **82.41%** 使用 `...pct_of_peak_sustained_elapsed`。naive 對應為 **99.38%／95.79%**。兩者分母不同，請讓文稿數字與截圖 section 一致，不要寫成資料矛盾。

## Part 3 — Shared memory 與 register blocking

### Q1：benchmark、實驗表與截圖

**本次已補測**：2048³，`all` 的十三列均 PASS。下表只比較題目要求的 smem16、smem32 與六組 reg2d；最快指的是這八組之中，不包含 Part 4 custom kernels。

先在終端顯示已保存的結果即可截圖，無須再量測：

```bash
cd /home/alen/2026-EdgeAI/lab1
cat results/adv_part3_benchmark_20261021.txt
```

截取 GPU／Problem／欄位名稱，以及 smem16 到第六組 reg2d 的全部八列。建議檔名 `part3_benchmark_2048.png`；圖說：「2048³ 的 shared-memory 與六組 register-blocking benchmark；皆為 20 次 launch 的中位數，correctness 全部 PASS。」

編譯資源可沿用 `pictures/make.png` 證明六組 reg2d；若要一併涵蓋 smem16、smem32，使用新編譯紀錄擷取相關區段，建議檔名 `part3_build_resources.png`：

```bash
sed -n '/Compiling entry/,+3p' results/adv_build_20261021.txt | c++filt
```

**建議文稿與實驗表：**

在 2048³ 下測試 smem16、smem32，以及六組改變 BM/BN、BK 與 TM/TN 的 reg2d。benchmark 時間取本次 20 次 launch 的中位數；registers/thread 與 static shared memory/block 來自目前 source 的 ptxas 編譯輸出。occupancy、warp stall 與 bank conflicts 來自既有、相同尺寸的 Nsight Compute 報告。為避免混淆，另列出 profiler 當時的 Duration。

| Kernel | 本次 benchmark（ms） | 舊 profiler（ms） | registers/thread | static smem/block（bytes） | 理論 occupancy | achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smem16 | 1.6753 | 1.921504 | 39 | 2,048 | 100.00% | 97.47% |
| smem32 | 2.1483 | 2.143328 | 40 | 8,192 | 66.67% | 66.51% |
| reg2d_128x128x8_8x8 | 0.8909 | 0.934944 | 94 | 8,192 | 33.33% | 25.07% |
| reg2d_64x64x8_4x4 | 0.7148 | 0.761600 | 40 | 4,096 | 100.00% | 99.05% |
| reg2d_128x64x8_8x4 | 0.9196 | 0.982496 | 66 | 6,144 | 50.00% | 49.74% |
| reg2d_64x128x8_8x4 | 0.9324 | 0.995712 | 66 | 6,144 | 50.00% | 49.63% |
| reg2d_64x64x16_4x4 | 0.7311 | 0.775680 | 40 | 8,192 | 100.00% | 98.19% |
| **reg2d_128x128x32_8x8** | **0.5673** | **0.649728** | **108** | **32,768** | **33.33%** | **26.81%** |

| Kernel | 主要 stall | cycles／issued instruction | Shared-load bank conflicts | Shared-store bank conflicts |
| --- | --- | ---: | ---: | ---: |
| smem16 | MIO Throttle | 21.01 | 19,695 | 0 |
| smem32 | MIO Throttle | 21.68 | 0 | 4,443,919 |
| reg2d_128x128x8_8x8 | Long Scoreboard | 7.65 | 33,593,130 | 14,680,064 |
| reg2d_64x64x8_4x4 | Long Scoreboard | 11.16 | 0 | 29,360,128 |
| reg2d_128x64x8_8x4 | Long Scoreboard | 11.24 | 0 | 29,360,128 |
| reg2d_64x128x8_8x4 | Long Scoreboard | 10.32 | 0 | 14,680,064 |
| reg2d_64x64x16_4x4 | Long Scoreboard | 10.69 | 0 | 62,914,560 |
| reg2d_128x128x32_8x8 | Long Scoreboard | 2.91 | 33,580,890 | 65,033,435 |

bank conflicts 是該次 launch 的計數，load 與 store 分開列出，不是百分比或幾路衝突。各組的存取寬度與 requests 數不盡相同，因此總數可指出問題，不能單獨當成每個 request 的效率排名。

**八組 profiler 截圖：**

上述每個 kernel 都對應 `reports/ncu_<完整 kernel 名稱>_2048_2048_2048.ncu-repz`，例如 `smem16` 對應 `ncu_smem16_2048_2048_2048.ncu-repz`。對八組逐一套用以下清單；若能清晰保留各欄值，可合併截圖。

| 建議命名規則 | GUI 位置 | 必須保留的內容／圖說 |
| --- | --- | --- |
| `part3_<kernel>_occupancy.png` | Occupancy；header／Summary 的 Duration | Theoretical Occupancy、Achieved Occupancy、Block Limit Registers／Shared Mem／Warps；圖說「`<kernel>`，2048³：理論與實測 occupancy 及資源限制。」 |
| `part3_<kernel>_warp.png` | Warp State Statistics → Warp State (All Cycles) | 上表主要 stall 與 cycles/instruction；圖說「`<kernel>`，2048³：主要 warp stall 與平均等待。」 |
| `part3_<kernel>_shared.png` | Memory Workload Analysis → Shared Memory table | **Loads／Stores 的 Bank Conflicts**；圖說「`<kernel>`，2048³：shared load/store bank-conflict 計數。」 |

Q1 表格內引用八組數據，截圖也應覆蓋八組；只放最快／最慢兩組不足以佐證整張實驗表。若數值顯示為 `33.59 M` 等縮寫，可另截 Raw 原值，見文末 metric 對照表。

**shared-memory 數字提醒：** 本表使用 ptxas 的 static allocation。舊 NCU 報告還列出 **1,024 bytes/block 的 driver shared memory**，例如 8,192 bytes static 加 driver 後，總量顯示為 9.216 Kbyte。不要將這個總量直接覆寫 static smem 欄；NCU 的 Kbyte 採十進位單位。

### Q2：每 FMA 的 shared-memory loads

**截圖需求：** 此題以程式與公式推導即可，不需要新增 profiler 截圖；效能關係可引用 Q1 benchmark。若加程式截圖，保留 smem 的 inner loop，以及 reg2d 載入 `ra`／`rb` 和 outer product 的區段即可。

**建議文稿：**

在 3a，一個 thread 計算一個 C 元素，每個 k 讀取一個 shared A 值與一個 shared B 值，完成一次 FMA，因此為 **2 個 FP32 shared-memory 元素載入／FMA**。

在 3b，每個 k 將 TM 個 A 值與 TN 個 B 值由 shared memory 載入 registers，再用 outer product 執行 TM×TN 次 FMA，所以

$$
\text{shared loads/FMA}=\frac{TM+TN}{TM\cdot TN}.
$$

當 TM=TN=8 時，16 個元素載入可服務 64 次 FMA，得到 **0.25 個元素載入／FMA**，相較 3a 減少 **8 倍**。其原因是同一個 A register 值重用於八個輸出欄，同一個 B register 值重用於八個輸出列。這裡計算的是 FP32 元素數，不是 SASS 載入指令數；編譯器可能使用 `LDS.128` 等一次讀取多個元素的指令。

本次 2048³ 中，最快的 reg2d 為 0.5673 ms，比 smem16 的 1.6753 ms 與 smem32 的 2.1483 ms 更快。register blocking 降低每次 FMA 對 shared memory 的需求，是合理的加速機制；但 8 倍載入量改善不等於 8 倍總時間改善，仍受 global tile 載入、同步、register pressure、bank conflicts 等因素影響。

### Q3：最快組態與較慢組態比較

**需截圖：** 沿用 Q1 的 `reg2d_128x128x32_8x8` 與 `reg2d_128x128x8_8x8` 三類截圖，另各補一張 **Compute Workload Analysis**，保留 FMA 管線對應的 `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed`：**26.20%／18.30%**。建議檔名 `part3_<kernel>_compute.png`，圖說「2048³、BK=32 與 BK=8 的 FMA 管線利用率比較。」

**建議文稿：**

在本次測試的兩組 smem 與六組 reg2d 中，2048³ 最快的是 **reg2d_128x128x32_8x8**，時間 **0.5673 ms**。以相同 BM=BN=128、TM=TN=8，但 BK=8 的 **reg2d_128x128x8_8x8** 作為較慢比較組，其時間為 **0.8909 ms**；BK=32 約快 **1.570×**。兩者每 block 均為 256 threads，差異集中在 BK。

既有、同為 2048³ 的 ncu 數據如下：

| 指標 | BK=8 | BK=32 |
| --- | ---: | ---: |
| registers/thread（編譯） | 94 | 108 |
| static shared memory/block（bytes） | 8,192 | 32,768 |
| Theoretical Occupancy | 33.33% | 33.33% |
| Achieved Occupancy | 25.07% | 26.81% |
| Block Limit Registers（blocks/SM） | 2 | 2 |
| Block Limit Shared Mem（blocks/SM） | 7 | 3 |
| Long Scoreboard（cycles／issued instruction） | 7.65 | 2.91 |
| Barrier（cycles／issued instruction） | 0.44 | 0.15 |
| FMA pipe utilization（elapsed） | 18.30% | 26.20% |
| Shared-load bank conflicts | 33,593,130 | 33,580,890 |
| Shared-store bank conflicts | 14,680,064 | 65,033,435 |

BK 從 8 增為 32，使 K=2048 的 tile 迭代由 256 次降至 64 次；每輪有兩個 `__syncthreads()`，因此每 block 的同步次數由 512 降至 128。BK=32 的 register 與 shared-memory 用量雖增加，但兩者都由 register 資源限制為每 SM 至多兩個 blocks，理論 occupancy 仍是 33.33%，並沒有進一步下降。

ncu 中 Long Scoreboard 從 7.65 降至 2.91 cycles／issued instruction、Barrier 從 0.44 降至 0.15，FMA pipe utilization 從 18.30% 升至 26.20%。這與較大的 BK 減少 tile 切換及同步成本、改善計算與等待比例的推論一致。Long Scoreboard 對應等待 L1TEX 操作的資料相依性，不能直接說成 shared-memory bank conflicts。[NVIDIA warp-stall 說明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)

BK=32 的 shared-store bank conflicts 實際更多，shared-load conflicts 則接近，因此此處不宜寫成「因 bank conflicts 消失而變快」。同樣地，其他 64×64 組態的 achieved occupancy 約 98～99%，仍慢於本組的 26.81%，說明 occupancy 高低不能單獨決定 SGEMM 效能。以上 profiler 解釋引用舊量測，支持機制判讀；本次的 1.570× 則由本次 benchmark 時間計算。

## Part 4 — float4 與 warp tiling

### Q1：需補截圖與證據狀態

目前 chain 為 `reg2d_128x128x8_8x8 → custom_vec4 → custom_vec4_warptiling`。`part4_swizzling_*` 是其他嘗試的舊紀錄，不應當成目前 chain 的結果。

| 建議檔名 | 操作／來源 | 必須顯示的內容 | 建議圖說 |
| --- | --- | --- | --- |
| `part4_benchmark_4096.png` | `cat results/adv_part4_benchmark_20261021.txt`；設定與 SUMMARY 可分兩張截取 | 4096³、三輪×20 reps、三個 kernel 的各輪時間、best time、相對前一步 speedup | 「2026-10-21 的 chain benchmark：float4 達標，warp tiling 此次未達 1.05×。」 |
| `part4_previous_grade.png` | `cat results/part4_warptiling_grade.txt` | correctness／sanitizer 摘要、三個 kernel 時間與 `Part 4 points: 10 / 10` | 「既有完整 grade 紀錄；與本次 benchmark 為不同批次。」 |
| `part4_resources.png` | `sed -n '/Compiling entry/,+3p' results/adv_build_20261021.txt \| c++filt` | 三個 chain kernels 的 registers、static smem、0 spill | 「目前 chain 的編譯資源：baseline 94 registers，兩個 custom 均 91；smem 均 8,192 bytes。」 |
| `part4_sass.png`（建議補） | `cat results/part4_warptiling_sass_summary.txt`；亦可離線查看暫存 binary 的 SASS | 兩個 custom 的 `LDG.E.128.CONSTANT`、`LDS.128`、`STS.128` 與索引指令差異 | 「靜態 SASS 顯示向量存取與額外索引指令；此圖不是動態效能 counters。」 |

**目前無法由既有資料補齊的截圖：** 兩個 custom kernel 的 Nsight Compute metrics。現有 `results/part4_warptiling_ncu.txt` 只有 `ERR_NVGPUCTRPERM`，`reports/` 也沒有這兩個 custom 的有效 report；錯誤畫面不能替代效能指標截圖。此限制應在分析中明示。

若日後手邊另有這三個 chain kernel **相同尺寸、相同條件**的有效報告，應補齊：Memory Workload Analysis 的 shared load/store Bank Conflicts 與 Global Loads requests/sectors；Warp State Statistics 的主要 stalls；Instruction Statistics／Source 的載入與索引指令執行量；Occupancy 的理論與實測值。此清單只指出欠缺的證據，本次不進行新的 profiling。

### 建議文稿：統一實驗結果

本部分以 `reg2d_128x128x8_8x8` 為基準，依序加入 float4 向量化載入與 warp tiling，每一步保留前一步的優化。兩步皆維持 FP32 CUDA-core 計算，未使用 Tensor Cores 或 TF32。

2026-10-21 在 RTX 5090 補測 4096³，每輪使用 harness 預設 L2 flush、warmup=3、20 次 launch 的中位數，再取三輪最佳值。九次測試的 correctness 均 PASS，結果如下：

| Kernel | 第 1 輪（ms） | 第 2 輪（ms） | 第 3 輪（ms） | 最佳中位數（ms） | 對前一步 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| reg2d_128x128x8_8x8 | 8.03942 | 8.35686 | 7.68614 | 7.68614 | — |
| custom_vec4 | 3.07098 | 3.07094 | 3.08019 | 3.07094 | 2.503× |
| custom_vec4_warptiling | 3.06483 | 3.64954 | 3.65466 | 3.06483 | 1.002× |

加速比由紀錄中的五位小數時間計算，分別為 `7.68614/3.07094=2.502862...`、`3.07094/3.06483=1.001994...`。本次僅補跑 benchmark，沒有執行完整 grade，也沒有重新執行 sanitizer。

既有完整 grade 的紀錄為 baseline **7.200 ms**、float4 **3.085 ms**、warp tiling **2.811 ms**；腳本回報加速比分別 **2.334×／1.098×**，結果為 **10/10**。本次則只重現 float4 的成功，warp tiling **未達 1.05× 門檻**。舊 grade 的加速比由腳本以內部較高精度時間計算，再輸出三位小數，因此用畫面已四捨五入的 ms 重算可能略有差異。

本次 warp tiling 的三輪中位數從 3.06483 至 3.65466 ms，顯示該次量測的跨輪變動；不能單憑此資料判定是時脈、背景工作或其他哪個因素造成。舊的 10/10 是歷史量測結果，本次 benchmark 不應再宣稱兩步都穩定通過成功門檻。依題目，未達標的步驟仍可藉測量與原因分析作為 failed optimization 討論；本文件不把它當成已取得新一輪完整評分。

### 建議文稿：float4 向量化載入

第一步保留 baseline 的 BM=BN=128、BK=8、TM=TN=8，以及 shared-memory layout、outer product 與 scalar C stores，僅把 A、B tile 的輸入搬移改為 float4：每個 thread 一次讀取四個連續 FP32 元素；A 拆為四個值寫入轉置的 As，B 可直接以 float4 寫入 Bs。float4 代表 128-bit 存取，不是低精度浮點運算。

本次 baseline 從 **7.68614 ms** 降至 **3.07094 ms**，加速約 **2.503×**，超過 1.05×。目前編譯結果與資源如下：

| Kernel | registers/thread | static shared memory/block | spill stores／loads |
| --- | ---: | ---: | ---: |
| reg2d_128x128x8_8x8 | 94 | 8,192 bytes | 0／0 bytes |
| custom_vec4 | 91 | 8,192 bytes | 0／0 bytes |
| custom_vec4_warptiling | 91 | 8,192 bytes | 0／0 bytes |

離線查看編譯後 SASS，可見 float4 版本的 `LDG.E.128.CONSTANT` 與 `STS.128`；baseline 對應的 global input load 則是 scalar 寬度。這支持向量存取已被編譯器實作。一次搬移四個元素可減少搬移迴圈的執行次數，但 A 的搬移分工與 shared-memory 寫入分布也隨之改變。因此，載入指令／索引成本降低與 shared-memory 存取變化都是可能機制，不能直接把全部加速歸因於 DRAM 頻寬或 bank conflicts 改善。

目前沒有 `custom_vec4` 的有效 counters 報告，以上效能原因屬程式與靜態 SASS 支持的推論，並非動態指令數或 bank-conflict 量測結論。registers 由 94 降至 91 本身也不能證明 occupancy 上升。

launcher 僅接受完整 tiles，並檢查 A、B 的 16-byte 對齊；其他形狀回傳 SKIPPED。既有完整 check／grade 紀錄支持其 correctness；本次補測另確認 2048³ 與 4096³ 的 PASS。

### 建議文稿：warp tiling

第二步在 float4 版本上改變 warp/lane 對輸出元素的分工，以及相應的 shared-memory 讀取與 C 寫回座標。維持 128×128 block tile、BK=8、256 threads/block、每 thread 64 個 FP32 accumulators，以及 float4 tile loads 與同步流程。

| 層級 | 分工 |
| --- | --- |
| Block | 128×128 outputs，8 個 warps |
| Warp | 32×64 outputs，block 內為 4×2 個 warp tiles |
| Lane | 4×8 的 lane 排列；每 lane 計算四個分散的 4×4 sub-tiles，共 64 outputs |

本次最佳時間由 float4 的 **3.07094 ms** 變為 **3.06483 ms**，約 **1.002×**，時間僅降低 **0.199%**，未達成功門檻；另兩輪 warp tiling 分別為 3.64954 與 3.65466 ms，比相應的 float4 更慢。因此，本次應將它描述為尚未重現舊 grade 加速幅度的優化嘗試。

從位址分工看，原版相鄰 lanes 的 B shared-memory 起點相差 8 個 floats；新版同一 lane row 內相差 4 個 floats，且 C 寫回座標跟著改變。這可能改變 shared-memory transactions 與輸出存取效率，但需要 counters 才能確認。兩版皆為 91 registers/thread、8,192 bytes static smem、0 spill，且均使用 `LDS.128`，資源用量降低不能作為加速理由。

靜態 SASS 中，float4／warp tiling 的 `IADD` 分別為 **4／10** 條，後者另有 **1 條 IADD3**；`LOP3.LUT` 分別為 **9／12** 條。這顯示新的座標分工伴隨更多索引相關指令，可能抵銷存取分布改善的好處，但這是靜態指令清單，並不等於動態執行次數或已證實的瓶頸。缺少同條件 profiler counters 時，只能將額外索引成本、存取分布變化與量測波動列為待驗證原因。

### Part 4 現有 report 的具體修訂

1. 將原本兩節各自的 benchmark 表整併為上面的「本次結果＋既有 grade」；刪除「目前僅完成第一項、5/10」這類過時進度描述。
2. 保留目前 chain 的 float4 與 warp tiling，其他未列入 chain 的優化紀錄不必混入主答案。
3. 原本「warp tiling 成功取得 5 分、總分 10/10」改成「既有完整 grade 曾回報 10/10；本次補測未重現第二步門檻」，並分別引用兩份紀錄。
4. 原本「腳本以未四捨五入時間算出」改成「腳本以內部較高精度時間算出」；grade 實際讀取 harness 的五位小數 ms，再顯示三位小數。
5. 可縮短 warp/lane 索引程式碼，保留分工表與效能分析即可。題目只要求回答 Questions，不需要把所有實作細節與測試過程放進 PDF。
6. `ERR_NVGPUCTRPERM` 只說明缺少 custom counters，不代表 Part 1～3 的既有 report 無效；避免一概寫成「沒有任何 ncu 數據」。

## 指標追溯與截圖核對

下列名稱可在 GUI 的 Raw 頁搜尋。也可用 CLI **離線讀取**既有報告，指令沒有附加執行檔，不會啟動 kernel：

```bash
ncu --import reports/ncu_naive_2048_2048_2048.ncu-repz --page details --section SpeedOfLight --section WarpStateStats
ncu --import reports/ncu_naive_2048_2048_2048.ncu-repz --page raw --csv
```

| 文稿／表格內容 | 原始 metric 或來源 |
| --- | --- |
| Compute (SM) Throughput | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| Memory Throughput（SOL 百分比） | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| L1/TEX（SOL，active） | `l1tex__throughput.avg.pct_of_peak_sustained_active` |
| L1/TEX（Memory Workload Analysis，elapsed） | `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` |
| L2 throughput | `lts__throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM throughput 百分比 | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| 實際 DRAM GB/s | `dram__bytes.sum.per_second`；注意原值單位 |
| L1／L2 hit rate（整體） | `l1tex__t_sector_hit_rate.pct`／`lts__t_sector_hit_rate.pct`；不是只計 global-load 的 hit rate |
| Global-load sectors/request | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` ÷ `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` |
| Profiler 時間 | `gpu__time_duration.sum`；原始有 ms／us，表格已統一成 ms |
| 理論 occupancy | `sm__maximum_warps_per_active_cycle_pct`；GUI 的 Theoretical Occupancy |
| 實際 occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers | ptxas／`launch__registers_per_thread` |
| Static／driver shared memory | `launch__shared_mem_per_block_static`／`launch__shared_mem_per_block_driver` |
| Shared-load bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` |
| Shared-store bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` |
| LG Throttle | `smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio` |
| Long Scoreboard | `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` |
| MIO Throttle | `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` |
| Barrier | `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` |
| 平均 warp cycles/instruction | `smsp__average_warp_latency_per_inst_issued.ratio` |
| Part 3 FMA pipe utilization | `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` |

warp stall 表格採 **Warp State (All Cycles)** 的 cycles／issued instruction，並非 Warp Stall Sampling 的 sample 百分比。Raw CSV 可能將這些 ratio 的單位標為 `inst`，本文按同版本 GUI section 的定義寫作 cycles／issued instruction；若選到 sampling 圖，不應期待同一個數值。

最後檢查：Part 1、2 的原有 benchmark 三張可沿用；Part 3 補上新 benchmark、編譯資源，以及實驗表引用的八組 counters 截圖；Part 4 補上本次 benchmark 與既有 grade 的獨立截圖，並保留 custom counters 尚缺的說明。完成截圖後，在 report 以實際圖片路徑插入，避免留下尚不存在的連結。
