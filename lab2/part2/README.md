# Part 2：壓縮實作練習框架

目前已完成 Q1～Q3 的量化／反量化、group layout、2–8 bit packing、逐層暫時反量化、儲存量統計、SVD 與執行入口。Case1／2 可選 RTN（直接 rounding，含 MSE clipping）或 GPTQ；Case3 先以 activation whitening SVD 減少約21%儲存，再量化因子至整體50%目標；Case4 使用3:4 SparseGPT剪枝、稀疏儲存與2-bit量化。**可執行不代表符合各 Case 的 CR／PPL 門檻，這份專案尚非完成的作業提交。**

2026-11-08 已核對 [HackMD 最新要求](https://hackmd.io/OAKquROlSLGr9E3-P7Nm8A?view)，原文同步於 [HW.md](../HW.md)。四個 Case 門檻不變，正式入口為 `StudentID_run.sh`。先前指南曾將 SVD ratio 寫為保留比例；依本次需求，`SVD Compression Ratio` 改顯示減少比例，JSON 另存 `retained_ratio`，兩種數值都保留。HW.md 未另列 SVD 公式，報告須清楚註明定義。

## 先確認框架

在專案根目錄，以既有本機環境執行（不下載模型或資料）：

```bash
source ~/.venvs/edgeai-5090/bin/activate
python -m unittest discover -s lab2/part2/tests -v
python lab2/part2/exercises.py --case 1
```

四個 Case 均已接通，可通過靜態 TODO 檢查。測試包含框架接線、實際數值核心、MSE 搜尋與儲存量檢查；其中固定替身只用來隔離接線，不能拿來做真正量化。測試通過不代表 PPL 已達標。

## 學習順序與介面

1. **P2-Q1**：`quantization.choose_qparams(groups, valid, bits)`。輸入 `[out,groups,G]`，回傳 `[out,groups,1]` 的 scale 與 zero point。想想 padding、全零與常數群組；輸出必須 finite，scale 必須正值。
2. **P2-Q2**：`quantize_values(...)`。回傳同 shape 的 unsigned integer codes，範圍 `[0,2**bits-1]`；padding 應表示零。思考 round／clamp 與 dtype。
3. **P2-Q3**：`dequantize_values(codes,scales,zero_points)`。回傳相同 group shape 的浮點重建值。用手算小矩陣核對零值、負值與端點，再執行 `python lab2/part2/check_numerics.py`。
4. **P2-C1／C2**：`compress.choose_layer_configs()`。回傳涵蓋所有目標層的 dict；每個值為 `QuantConfig(bits,group_size)` 或 `None`。傳入 `--wbits` 時必須實際用於配置；未指定時由你決定策略，框架沒有隱藏的預設答案。
5. **P2-SVD／C3**：`svd_linear()` 建立兩個 Linear；`case3()` 預設使用 train activation whitening，依 `--svd-rate 0.21` 分配 rank，再量化部分因子列；`--svd-method plain` 保留普通 SVD 對照。`storage.svd.compression_ratio = 1-factor_bytes/original_bytes`，`retained_ratio = factor_bytes/original_bytes`，量測時點在因子量化前，分母涵蓋全部目標 W、排除 bias。最終 overall CR 另計所有實際儲存。
6. **P2-PRUNE／C4**：`prune_linear()` 聯合處理剪枝與量化，`case4()` 沿用逐層 train 校正。預設每4個權重剪除1個，以3個存活值及2-bit位置儲存；另採2-bit／group64、FP16 scale與packed zero point。`Pruning Rate` 依量化前稀疏表示的實際bytes計算，`Pruned Elements` 另記本次剪除的原始非零權重比例。

函式 docstring 提供題目對應、shape、提示及檢查項目。`replace_module()` 可替換原始目標，`PackedLinear.from_linear()` 可用於你建立的 Linear factors。每個 Case 必須從原始模型開始。

## 儲存與推論規則

- Overall CR 的分母固定為 `model.layers.*` 中 q/k/v/o、gate/up/down 七種 projection 的原始 weight+bias bytes；排除 `lm_head`。`None` 的目標層仍計入分子與分母。Overall CR=`1-final_bytes/original_layer_bytes`；SVD ratio 另用原始 W 帳本，兩者不可混用。
- codes 以低位元優先裝入連續 uint8 stream，支援跨 byte 邊界。最後一個 byte 的空位補零；每列 weight 沿 input features 分組，最後一群補零。
- 分子計入實際 codes、scale、zero point、bias、padding、32-byte shape/bits/group 設定 buffer，以及 replacement 內所有 Parameter/buffer。新增在目標層以外的 tensor 也計入。相同 tensor 物件在同一 target 中只計一次；不同 target 的運算分別計量。不同 view 的計量採各自 tensor bytes，非底層 storage 去重。
- `common.record_base()` 只能在原始模型上呼叫一次；`compress_model()` 已處理。`common.storage_report()` 提供逐層帳本。必要資料必須註冊為 Parameter/buffer，不能藏在一般 Python attribute 裡逃過統計。
- forward 每次只重建目前層的 weight，不保留浮點副本。這是可移植的 inference 路徑，可能較慢；不宣稱低位元硬體加速。使用後暫存離開作用域，CUDA allocator 仍可能保留可重用的 memory。
- `state_dict` 保存 packed buffers，但不自動重建 module 結構；重載前需建立相同配置的 replacement。基準帳本是當次程序的統計，不是 checkpoint 的一部分。
- `--rate` 是目標比例，絕不當成量測結果。純 4-bit／16-bit 的理想 CR 只有 75%，加上 overhead 會更低，不能直接通過 Case1 的嚴格 `>75%`。

### 單層 SVD：兩個 Linear factors

`compress.svd_linear(linear, rank)` 使用 PyTorch `torch.linalg.svd(..., full_matrices=False)`。對 `W[O,I]`，取 `B=Vh[:r,:]` 與 `A=U[:,:r]*S[:r]`，回傳 `nn.Sequential(first, second)`：`first.weight` 為 `[r,I]` 的 B，`second.weight` 為 `[O,r]` 的 A。forward 直接計算兩個較小的 Linear，不重建完整 W；原始 bias 只複製至第二層。

分解使用 FP32（FP64 來源保留 FP64 計算），factors 儲存為原 weight 的 dtype／device；參數凍結，保留原 training mode。factors 使用獨立緊密配置的 Parameter，不是完整 U／Vh 的 view，也不保存原始 Linear、S 或 dense W。實際 bytes 為 `r*(I+O)*element_size`，若原層有 bias 再加原 bias bytes，現有 `storage_report()` 可直接統計。

在 Part2 目錄檢查單層介面：

```python
import torch
from torch import nn
from compress import svd_linear

original = nn.Linear(6, 8).eval()
factors = svd_linear(original, rank=2)
x = torch.randn(2, 3, 6)
print(factors[0].weight.shape, factors[1].weight.shape)  # [2,6], [8,2]
print(factors(x).shape)                                # [2,3,8]
```

rank 必須為 `[1,min(O,I)]` 的整數。保留 full rank 是為了檢查重建，此時未必節省儲存。單層 helper 的重建測試不代表 Case3 的 CR／PPL 已達標。`lm_head` 仍排除於目標範圍之外。

### Case3：先 SVD 減少21%，再量化因子

流程為原始 W → train activation whitening SVD → 兩個浮點因子 → `MixedPrecisionLinear`。`--svd-method whiten` 是預設，`--svd-method plain` 使用原本的 `svd_linear()`。SVD 的 rank 依 `--svd-rate`（預設0.21，表示減少21%、保留79%）規劃，與最終 `--rate 0.5` 分開。各層先取比例預算的 floor rank，再以 floor／ceil 的總 bytes 修正；量化前的因子大小記為 SVD-only 階段資料。

Whitening 採 [SVD-LLM §3.1](https://arxiv.org/html/2403.07378v5#S3.SS1) 的截斷方式：收集輸入列向量的 `G=mean(x*x.T)`，以 Cholesky 得到 `L*L.T=G+jitter*I`，對 `W*L` 做 SVD。兩因子為 `A=U_r*sqrt(S_r)`、`B=sqrt(S_r)*Vh_r*inv(L)`；實作用 triangular solve 計算 B，不建立 L 的逆矩陣。這個版本只採 whitening，不包含論文後續的 LoRA／參數訓練，因此不是完整 SVD-LLM 重現。

`svd_whitening.py` 重用現有 WikiText-2 **train** 取樣及 Qwen block-input 捕捉流程。預設 `--svd-samples 128 --svd-seqlen 2048 --svd-damp 0.0001`，seed 使用 `--seed`。q/k/v 共用統計，gate/up 共用統計；o、gate/up、down 依序重新收集輸入，包含先前群組已壓縮的影響。每個 block 處理後，將輸出傳至下一 block。固定大小的 block 輸入與當前 Gram／Cholesky 都留在模型所在裝置；方法沒有權重 offloading、KV 壓縮或梯度更新。

Damping 是 Gram 對角線平均乘 `--svd-damp`；只有 Cholesky 實際失敗時才逐次增加 jitter，實際值逐層記錄。校正窗口長度不改變官方 PPL 的2048-token評估切段。所有校正資料、Gram、whitening 矩陣與完整 SVD 暫存於轉換後釋放，最終只保留兩因子與必要量化資料。

因子量化使用 RTN，預設8-bit、group128、MSE qparams。`--wbits` 決定低位元部分的位寬，**不表示全部因子都必須量化**。規劃器混用低位元尾端列與原 dtype 的前端列，先依共同儲存預算分配，再用下一批列的 subset sum 修正。預設 float16 下便是8-bit／FP16混合。第一個因子的前端列對應較大的奇異值方向；第二個因子的列是輸出 channels，目前沒有額外做敏感度校正。

`MixedPrecisionLinear` 保存獨立緊密配置的浮點列與 PackedLinear，forward 將兩部分輸出按列拼接、最後加 bias；不保留原始 dense W 或完整浮點因子副本。bias 只保存一次。實際 bytes 包含 codes、padding、FP32 scale／zero point、每個 packed child 的32-byte設定與所有未量化列。為了讓非整除 bits／group size 的計畫與 packing 完全一致，量化列數以 byte 對齊的小單位配置。

```bash
cd lab2/part2
mkdir -p results
python run_eval.py --case 3 --rate 0.5 --svd-rate 0.21 \
  --svd-method whiten --svd-samples 128 --svd-seqlen 2048 --svd-damp 0.0001 \
  --wbits 8 --group-size 128 --qparam-method mse \
  --metrics-json results/case3_whiten_svd21_quant8_g128.json
```

`case3_details` 記錄 `order=svd_quantize_factors`、`svd_method`、`rank_budget`、`quantization_budget`、每層 rank／位寬／量化列數。`whitening` 另存演算法、校正設定、train fingerprint、窗口起點、token hash、耗時；每層另存實際 jitter、校正 token 數、丟棄的 regularized weighted singular-value energy 比例。這些能追溯此次校正，不能代替官方 PPL。後者含 `target_difference_bytes` 與 `exact_target`：對小矩陣、不同 group size 或不可達的整數預算，仍回報實際 bytes，不把目標值當成實測值，也不新增無用途 buffer 湊大小。`storage.svd.factor_bytes` 是量化前的浮點權重因子大小；`storage.compressed_bytes` 才是最終模型表示。

本機 **RTX 5090／Python 3.11／torch 2.7.0+cu128** 已以完整 Qwen3-4B（float16、seed0）執行上述命令並沿用官方 PPL 程序。結果為 **PPL 20.3553925014、overall CR 50.000000%、SVD CR 21.000011%**，符合本機 Case3 三項門檻；T4／torch2.5.1+cu121 尚未實機驗證。

| 項目 | 實測值 |
| --- | ---: |
| 原始目標 W bytes（排除 lm_head） | 7,266,631,680 |
| SVD-only 浮點因子 bytes | 5,740,638,208 |
| 最終表示 bytes（含所有量化必要資料） | 3,633,315,840 |
| 額外目標層外 tensor bytes | 0 |
| 官方 PPL | 20.3553925014 |
| 校正＋轉換時間（不含載入／PPL） | 272.18 秒 |

詳細帳本在 `results/case3_whiten_svd21_quant8_g128.json`，來源 hashes 與命令在同名 `.provenance.json`，stdout／校正日誌在同名 `.stdout.txt`／`.log`。校正使用 WikiText-2 **train** 的128×2048 token窗口，damp=0.0001；token SHA256 為 `ec0a043b2c48080ebb5cce82773928c13562007527fc443b1fd8f651a02820d8`，逐層 jitter 和 weighted energy 亦記錄在 JSON。43項本機測試通過，包含 CPU／CUDA 的加權誤差已知解、完整 rank 重建、實際 Qwen 小模型逐層校正接線、儲存量與 CLI。這些實測可 confirm 此配置的結果；後續若更改校正、rank 或精度，要重新量測以支持其 logic。

只隔離 SVD、不量化因子的診斷命令：

```bash
python run_eval.py --case 3 --rate 0.21 --svd-rate 0.21 \
  --svd-method whiten --metrics-json results/case3_whiten_svd_only.json
```

此 Qwen 配置的量化列數為0，整體 CR 約21%，用途是分辨 SVD 與因子量化的影響。普通 SVD 對照改用 `--svd-method plain`；兩種方法須固定相同原始模型、dtype、seed 與官方評估程序。普通 SVD 的校正參數不會使用。

#### 舊版實驗紀錄（不同流程，不能當作新版結果）

使用者回報普通 SVD-only 的 CR **21.000011%**、PPL **1,318,900.2766**（`case3_svd_only.json`，量化列數0）；普通 SVD加因子量化的回報為 CR **50%**、PPL **1,273,823.6550**。這些是 whitening 前的對照，不是新版結果。

舊版 `quantize_dequantize_svd` 先量化再反量化，最終只存 FP16 因子，量化的儲存收益沒有保留。本機5090／torch2.7.0、8-bit／group128／MSE 的舊結果為 CR **50.000000%**、PPL **6,530,978.0408**；資料在 `results/case3_quant8_svd_g128.json` 與 `results/case3_quant8_svd_provenance.json`。同設定僅量化的對照為 CR **46.874889%**、PPL **13.7436**，見 `results/diagnostic_rtn8_g128.json`。這些檔案與來源 hashes 保留原樣，新版必須使用不同檔名重新實驗。

### MSE clipping：相同儲存量下選擇量化範圍

原本的 `choose_qparams()` 保留為 min/max 對照。`choose_qparams_mse()` 使用純 PyTorch，對每個 group 測試零值包含在內的範圍：從原 min/max 的 100% 到 50%，共21個候選。每個候選都使用目前 Q2／Q3 的 rounding、clamp 與反量化，對**原始有效權重**計算平方誤差，包含被裁切的權重，排除 padding。保留最小誤差的 scale／zero point；原 min/max 一定列入比較，相同誤差保留較寬範圍。

搜尋只發生在模型轉換階段，對 group 分塊、對候選向量化，限制 FP32 暫存用量。最後仍只儲存原格式的 codes、FP32 scale、FP32 zero point、bias 與設定 buffer；不保存額外 clipping thresholds、mask 或原始浮點權重。因此在相同 bits、group size 與層範圍下，CR 與原 min/max 相同。權重 MSE 不劣於原方法，**PPL 不保證一定降低或通過15**，需以官方程序實測。

`QuantConfig` 的 `qparam_method` 預設為 `mse`；也可指定 `minmax`。CLI 的 `--qparam-method`、`--mse-grid-size`（預設21）、`--mse-min-ratio`（預設0.5）覆寫此次轉換設定，並寫入 metrics JSON。`--mse-grid-size 1` 等同保留原 min/max。搜尋不使用校正資料或更新梯度；這是 Part2 實驗，不能直接當作 Part3 AdaptQuant 重現。

目前未指定 `--wbits` 時，Case1 使用 attention4-bit／MLP3-bit，Case2 全部使用3-bit；指定時統一覆寫為該位元數。Case2預設group512，其餘設定仍可透過 `--group-size` 調整。轉換進度會顯示在 stderr，stdout 仍只印最後的 CR／PPL。

在 Part2 目錄，用相同原始模型／dtype／seed 對照：

```bash
python run_eval.py --case 1 --rate 0.75 --group-size 128 \
  --qparam-method minmax --metrics-json case1_minmax.json
python run_eval.py --case 1 --rate 0.75 --group-size 128 \
  --qparam-method mse --metrics-json case1_mse.json
```

兩個命令都使用目前的4／3-bit分層配置。若要重現舊的全部3-bit實驗，在兩個命令都加 `--wbits 3`。兩次執行須使用同一份原始模型。新方法不需額外安裝第三方量化套件，既有 requirements 即可。

本機5090、Python3.11.18、torch2.7.0+cu128、float16、seed0，以官方入口完整評估的結果：

| 方法 | attention／MLP bits | group size | CR | PPL | Case1 |
| --- | --- | --- | --- | --- | --- |
| min/max | 4／3 | 128 | 76.501512% | 21.6025 | PPL未通過 |
| MSE clipping，21候選 | 4／3 | 128 | 76.501512% | 19.4685 | PPL未通過 |
| GPTQ，min/max，train校正128×2048 | 4／3 | 128 | 76.501512% | 14.6997 | 本機通過 |

min/max 與 MSE 的對照已確認（confirm）252個目標層的原始／壓縮 bytes 全部相同，額外 tensor bytes 為0；比較的推理（logic）是固定儲存配置，只更換範圍選取方式。這組對照的 PPL 約降低2.1340，單獨使用 MSE 仍未通過 Case1。完整數據在 `results/case1_mixed_g128_minmax.json`、`results/case1_mixed_g128_mse21.json`，對照摘要與程式 hashes 在 `results/case1_qparam_comparison.json`。這些是本機結果，T4／指定torch2.5.1環境仍需重跑。

2026-11-09 加入 GPTQ，使用同一原始模型／float16／seed0／4與3-bit配置／group128，由官方入口取得 PPL **14.699745283829614**、CR **76.50151240361202%**。其原始 bytes **7,266,631,680**、壓縮 bytes **1,707,548,544** 與上述兩種方法逐層相同，沒有額外 tensor。GPTQ 量化階段約274.2秒；此時間不包含模型載入與 PPL 評估。這組結果在本機符合 Case1 的 CR>75%、PPL<15，**尚未在 T4／torch2.5.1+cu121 確認**。原始結果、比較與 source hashes 分別在 `results/case1_mixed_g128_gptq128.json`、`results/case1_gptq_comparison.json`、`results/case1_gptq_provenance.json`。數值不可沿用至不同校正設定或其他 GPU。31項測試通過，包含 CPU／CUDA 的獨立公式核對、逐層校正、padding、相同儲存量與錯誤處理；測試日誌為 `results/gptq_framework_tests.log`。

### GPTQ：用校正輸入補償量化誤差

`gptq.py` 依 [GPTQ 論文 Algorithm 1](https://arxiv.org/abs/2210.17323) 實作，並參照[作者的實作](https://github.com/IST-DASLab/gptq)。只使用既有 PyTorch／datasets／transformers 依賴，不需安裝第三方量化 kernel。校正目標是減少 Linear 的輸出誤差；這與只比較原始權重誤差的 MSE clipping 不同。

- 對每個 projection 的輸入累加 `H = 2/N * X.T @ X`，`N` 是 token 數。對角線加上 `0.01 * mean(diag(H))`，再以 `torch.linalg.cholesky` 建立逆 Hessian 的上三角因子。量化一個 column 後，將誤差補償至尚未量化的 columns；每128個 columns 做一次尾部矩陣更新，並在 group 邊界切開 block。
- 使用 **WikiText-2 raw train split**，seed0 隨機取128個長度2048的完整窗口。從不讀 test split 做校正，也不使用 loss 或梯度更新。官方 PPL 仍由 `evaluate.py` 對 test split、2048切段計算。
- 逐 decoder block，依序處理 q/k/v → o → gate/up → down。q/k/v 共用輸入統計，gate/up 也共用；每組替換後重新執行 block，讓後續組與後續 block 的校正包含上游量化誤差。實作限定目前的 Qwen3 full-attention decoder。
- 使用原本的 column 順序與動態連續分組，未啟用 activation order。每組 scale／zero point 由已補償的權重選取，因此不需另存 permutation 或 group index。`--qparam-method minmax` 為 GPTQ 預設；也可明確指定 `mse`，但兩者是不同實驗。
- 最終仍存為 `PackedLinear`，其 buffers、dtype、padding、bias 與 shape metadata 大小完全沿用原格式。同 bits／group size／層範圍的 CR 不增加；Hessian 與校正 activation 在轉換完成後釋放，權重不 offload、不長期保存浮點副本。轉換時整個模型與一份 block-input activation buffer 留在同一裝置。T4 的實際峰值與相容性仍需實機確認。
- 無校正資料、非法參數或 Cholesky 失敗會明確停止，不會退回 RTN 或原始模型。錯誤訊息含 damping 調整提示。

在已啟用環境的 **Part2 目錄**，重現本輪 Case1 實驗：

```bash
mkdir -p results
python run_eval.py --case 1 --rate 0.75 --quantizer gptq \
  --group-size 128 --qparam-method minmax \
  --gptq-samples 128 --gptq-seqlen 2048 --gptq-damp 0.01 \
  --metrics-json results/case1_mixed_g128_gptq128.json
```

未指定 `--wbits`，因此沿用 attention4／MLP3-bit。`--gptq-block-size` 預設128；它是誤差更新的計算區塊，與量化的 `--group-size` 不同。`--gptq-seqlen` 只控制校正窗口，不改官方評估長度。要先檢查流程可減少 `--gptq-samples`，但不同校正設定須分開記錄，不能沿用正式實驗的 PPL。

`--quantizer rtn --qparam-method minmax` 可重跑上表的 min/max 對照；直接呼叫 `run_eval.py` 時，Case1 不指定 quantizer 使用 RTN＋MSE，Case2 預設 GPTQ＋min/max。完整 `run.sh` 的 Case1／2 預設 GPTQ，Case3 使用 RTN，Case4 使用 SparseGPT；可用環境變數 `CASE1_QUANTIZER=rtn` 或 `CASE2_QUANTIZER=rtn` 切換對照。

### Case2：3-bit GPTQ與精簡量化參數

所有decoder的q/k/v/o、gate/up/down使用3-bit GPTQ，預設group512、min/max、WikiText-2 **train** 校正128×2048、damp0.01、seed0。沿用既有逐層校正與誤差補償，不做剪枝或fine-tuning。

`QuantConfig.compact=True` 讓RTN與GPTQ共用 `CompactPackedLinear`：scale儲存為FP16，zero point以3-bit打包。兩種方法都先將scale捨入至實際儲存精度，再決定codes；不長期保存浮點權重。`case2()`保留RTN對照入口，正式CLI及學號腳本預設選GPTQ。

FP16基準下，每個權重的平均儲存位寬為 `3+(16+3)/512=3.037109375 bits`，理論CR為 **81.01806640625%**。Qwen3-4B的目標輸入維度皆可被512整除；實際CR仍計入每層32-byte metadata及bias。較小group或較高bits可能超出Case2的儲存預算，不能沿用這個CR。

```bash
mkdir -p results
python run_eval.py --case 2 --wbits 3 --group-size 512 \
  --quantizer gptq --qparam-method minmax \
  --gptq-samples 128 --gptq-seqlen 2048 --gptq-damp 0.01 \
  --metrics-json results/case2_gptq3_g512.json
```

不指定上述壓縮參數時，`python run_eval.py --case 2`也使用同一配置。結果、逐層儲存帳本及校正資訊寫入metrics JSON；本次實驗的來源hashes另存於同名 `.provenance.json`。

2026-11-09，本機RTX 5090／Python3.11.18／torch2.7.0+cu128，以同一原始Qwen3-4B、float16、seed0，經官方 `run_eval.py` 評估：

| 項目 | 實測值 |
| --- | ---: |
| 原始目標層bytes | 7,266,631,680 |
| 最終壓縮bytes | 1,379,355,264 |
| 額外目標層外tensor bytes | 0 |
| Overall CR | 81.0179554332% |
| 官方PPL | 16.0910335961 |
| 校正與轉換時間（不含載入及PPL） | 272.55秒 |

252個目標層均使用 `CompactPackedLinear`，本機結果同時符合CR≥81%與PPL<20。43項既有測試通過。原始結果為 `results/case2_gptq3_g512.json`，stdout、日誌與來源hashes分別在同名 `.stdout.txt`、`.log`、`.provenance.json`。這些資料可確認（confirm）本次結果；方法的推理（logic）是縮小量化參數的儲存成本，讓所有目標權重保留3-bit，再由GPTQ補償量化誤差。此結果尚未在T4／torch2.5.1+cu121重跑。

### Case4：3:4 SparseGPT剪枝與2-bit量化

依助教QA，剪枝本身須減少至少20%的實際儲存。只把權重設為零並保存完整codes的舊方法不符合此條件。

沿用GPTQ的WikiText-2 **train** 校正與 q/k/v → o → gate/up → down 順序。在每個量化group中，將每列連續4個權重分成一組，以 `w² / R[j,j]²` 選擇移除成本最低的1個位置；`R` 為逆Hessian的上三角Cholesky因子。每個column同時處理剪枝與量化，再補償後續columns的誤差。被剪除的位置固定為zero-point code。

`Sparse34PackedLinear` 只儲存每4個權重中的3個存活codes及2-bit剪除位置；不儲存被剪除位置的code。另存每組FP16 scale、packed zero point與32-byte metadata。推論時暫時重建零值，不保存dense權重副本。input width與group size須為4的倍數。

剪枝單獨使用原始FP16精度時，每4個權重由64 bits減至 `3*16+2=50 bits`，理論儲存縮減為 **21.875%**。`storage.pruning.original_bytes`／`compressed_bytes` 計入原始dtype的存活值、實際packed位置bytes、32-byte metadata及bias；`rate`／`compression_ratio` 由這份量化前帳本計算。`element_fraction` 另記本次剪除的原始非零權重比例，不拿它代替儲存縮減。

接上2-bit／group64量化後，每個原始權重平均使用 `(3*2+2)/4+(16+2)/64=2.28125 bits`，整體理論CR仍為 **85.7421875%**，實際CR包含metadata、bias與padding。四個Case的PPL均使用原本官方程序。

新版完整模型的實測結果另存 `results/case4_sparse34_g64.json`；舊 `results/case4.json` 與 `pic/case4.png` 屬於20%非結構化剪枝，不能作為新版的結果。

2026-11-11，完整Qwen3-4B在本機RTX 5090／Python 3.11／torch 2.7.0+cu128，使用官方PPL程序得到：

| 項目 | 實測值 |
| --- | ---: |
| 原始目標層bytes | 7,266,631,680 |
| 量化前稀疏表示bytes（含位置索引、metadata） | 5,677,064,064 |
| Pruning Compression Ratio | 21.874889% |
| 剪除元素比例 | 25.000000% |
| 最終表示bytes | 1,036,070,784 |
| Overall CR | 85.742077% |
| 官方PPL | 41.3808761065 |

252層皆使用 `Sparse34PackedLinear`，每層的pruning-only CR皆≥20%，額外目標層外tensor bytes為0。47項CPU／CUDA測試通過，包含獨立dense重建、padding、block邊界、序列化、量化前稀疏表示的實際儲存與Qwen逐層校正接線。完整校正與轉換耗時約243.93秒，不含載入與PPL評估；T4／torch 2.5.1+cu121尚未實機驗證。


在Part2目錄執行：

```bash
mkdir -p results
python run_eval.py --case 4 --wbits 2 --group-size 64 \
  --qparam-method minmax \
  --gptq-samples 128 --gptq-seqlen 2048 --gptq-damp 0.01 \
  --metrics-json results/case4_sparse34_g64.json
```

metrics JSON 除逐層 bytes 外，另存 `gptq.settings`、train split 來源／fingerprint、抽樣起點、選用 token IDs 的 SHA256、每層 dead columns、實際 damping 與二次型誤差估計。這些統計是一般數字／文字，不是推論所需 tensor。請一併保存命令、模型來源、GPU／套件版本、日誌與程式 hash；只有官方 PPL 能 confirm 最終品質，Hessian 誤差估計不能代替 PPL 的 logic。

GPTQ 使用目標 Qwen 的校正輸入，**只適用這次 Part2 實驗，不符合 Part3 的 target-model calibration-free 限制**。Part3 程式未接入 GPTQ。

### 解包與推論效能

`unpack_codes()` 已改為以 byte 區塊直接取出 codes；2／4-bit 使用單 byte 位移，3／5／6／7-bit 最多合併兩個相鄰 byte，避免逐 bit 掃描與每個元素的 int64 索引。packed 格式、qparams、bias 與設定 buffer 都沿用原本表示，未新增持久化 tensor 或浮點權重快取。

依本次要求，Part2 壓縮入口、SVD、packing／quantization、GPTQ 的手寫輸入判定已移除，`PackedLinear.validate()` 與 `validate_on_forward` 也已移除。參數需由呼叫者提供正確值；底層 PyTorch／Python 自身的錯誤、GPTQ 實際分解失敗與未完成 TODO 的錯誤保留。數值檢查改由測試與 `check_numerics.py` 執行。官方 PPL 評估程序維持原樣。

從 Part2 目錄量測一個合成 Linear（不下載 Qwen）：

```bash
python bench_inference.py --bits 3 --group-size 128 --output results/bench_3bit_g128.json
python bench_inference.py --bits 3 --group-size 96 --output results/bench_3bit_g96.json
```

比較使用相同的 packed buffers 與相同浮點矩陣乘法，執行前確認原版／新版 codes 與單層輸出逐值完全一致，並檢查持久化 bytes 沒有增加。輸出包含同步後的單層 latency 中位數、加速比與 PyTorch allocator 的峰值暫存用量；這些數字不代表完整 Qwen 的端到端加速比或正式 PPL。其他 GPU 程序會影響時間，請在同一環境比較。新版仍每次暫時反量化，目前沒有引入額外依賴或 Blackwell 專用 kernel，T4／torch2.5.1 的實機相容性仍需驗證。

已啟動的 Python 程序不會自動載入檔案修改；下一次以原本的 `run_eval.py` 命令啟動時才會使用新版。不要為量測速度變更評估資料或切段。

## 環境與執行

正式環境：Python 3.11、torch 2.5.1+cu121。`requirements.txt` 使用指南中既有的 transformers 4.56.2、datasets 3.6.0、accelerate 1.10.1；本次僅在本機 Python 3.11／torch 2.7.0+cu128 驗證框架，T4 相容性尚待確認。

單獨執行 `run_eval.py` 時，除 CR 與 PPL 外也會顯示 `SVD Compression Ratio` 和 `Pruning Rate`。未記錄的指標顯示 `N/A`，已記錄的零值顯示 `0.000000%`；`--metrics-json` 的 `storage.svd`／`storage.pruning` 保留原始計數和定義。`Pruning Rate` 顯示量化前的實際儲存縮減，`Pruned Elements` 另顯示剪除元素比例。SVD 顯示量化前的減少比例 `1-factors/original_W`；JSON 另存原本的保留比例 `retained_ratio`。

在 T4 的乾淨 Python 3.11 環境，完整入口會安裝依賴並依次執行四個 Case：

```bash
cd lab2/part2
bash 314552017_run.sh
```

腳本依次從原始 `Qwen/Qwen3-4B` 載入並執行四個Case，各次載入前皆先檢查TODO。套件安裝與診斷在 stderr；stdout 只顯示每個 Case 的 CR 與 PPL。每個 Case 的 byte ledger、設定與模型來源 存於 `results/caseN.json`。若遇錯誤，腳本立即停止，不能將先前成功的 Case 視為全部完成。

本機 5090 已建立環境時，避免安裝 T4 的 torch：

```bash
SKIP_INSTALL=1 bash 314552017_run.sh
```

單獨測試與 baseline（從 Part2 目錄）：

```bash
python run_eval.py --dtype float16 --seed 0
python run_eval.py --case 1 --rate 0.76 --wbits 3 --group-size 128 --metrics-json case1.json
```

後一組參數僅示範 CLI，**不是已驗證的達標策略**。Case2預設group size 512、Case4預設64，其餘預設128；bits與分層設定仍可調整。PPL 仍使用原本的 WikiText-2 test、2048 切段與 loss，未調整評估程序。模型與資料會從 Hugging Face 下載；四個 Case 需使用同一份原始模型，請保留 metrics。

## 報告需要保存什麼

| Case | 方法與各層設定 | 原始 bytes | 壓縮 bytes | CR | SVD CR／剪枝率 | 官方 PPL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 待填 | | | | 不適用 | |
| 2 | 待填 | | | | 不適用 | |
| 3 | 待填 | | | | | |
| 4 | 待填 | | | | | |

必要數據是四個 Case 的 CR／PPL、Case3 的 SVD-only bytes、Case4 的量化前稀疏bytes、位置metadata成本及剪枝計數。建議圖片為 CR–PPL 圖，以及只改一個設定的比較圖。額外保存 GPU、Python／套件版本、程式 hash、執行命令；未量測的欄位留空。

用實測確認（confirm）結果，交代設定與結果間的推理（logic）。比較 min/max 與 MSE 時，保存相同配置的 CR／PPL，區分權重重建誤差與模型品質，不能用前者代替後者。Case3 的 overall CR 在 HW.md 寫成 `=50%`；指南提及的 ±0.5 百分點不是 HW.md 明列規則，需有助教確認才能採用。結果僅在列出的環境有效，本次不產生報告 PDF。
