# Part 1：Quantization 學習引導

依據：[作業要求](../HW.md)、[題目 notebook](part1.ipynb)。本文件提供推導方向、操作順序與檢查方法，填空由你完成。

## 1. 先確認環境與修改範圍

- Part 1 占 30%，使用 **PyTorch 2.6.0**；與 Part 2、Part 3 的環境分開。
- 使用 GPU runtime。notebook 會把 MobileNetV2 移到 CUDA，作業列出的 T4 環境可作為起點。
- 依 HW.md 的連結下載 `mobilenetv2_0.963.pth`，確認 notebook 預期的檔案位置。
- 只修改 `YOUR CODE STARTS HERE` 到 `YOUR CODE ENDS HERE` 之間的內容；不要修改測試或給定的 helper。

目前 setup cell 的安裝指令沒有鎖定版本，直接執行不保證得到 2.6.0。先在 notebook 外部的環境設定中處理版本，再用 notebook 已有的 `print(torch.__version__)` 確認。模型路徑目前寫成 Kaggle 路徑；若使用其他平台，先詢問助教允許的路徑調整方式，不要為了通過測試改寫受限制的程式區塊。

開始前記下 Python、PyTorch、GPU 型號與模型檔案來源。完成後重新啟動 kernel，由上到下執行，避免殘留變數讓錯誤被掩蓋。

## 2. 先建立三個觀念

1. **量化與反量化**：作業使用 `r = S(q - Z)`。`r` 是實數、`q` 是量化整數，`S` 是 scale，`Z` 是 zero point。反量化通常只能近似還原原值。
2. **數值範圍與儲存型別不同**：2-bit 數值可以暫存在 `int8` tensor，但這不代表每個值實際只占 2 bits。這個差別會直接影響 Part 2 的壓縮率。
3. **broadcasting 必須對到正確軸**：per-channel scale 應對應 output channel；Linear 與 Conv2d 的輸出 shape 不同，不能只因程式可執行就認為正確。

建議順序：Q1 → Q2 → Q3 → Q4 → Q5 → Q6，確認前段測試與觀察結果後，再做 Q7 → Q8 → Q9 → Q10。

## 3. Q1：單一 tensor 的線性量化

位置：`linear_quantize()`。

先在紙上從作業的反量化關係整理出量化方向，再依 notebook 的既有流程辨認：縮放、四捨五入、型別轉換、zero point 位移、範圍限制。分清楚哪些步驟已由題目提供，只完成空格。

動手前想一想：

- 若直接轉成整數而未四捨五入，負數會發生什麼事？
- 實數 0 應對應哪個量化整數？
- signed 2-bit 能表示哪些整數？超出範圍後應如何處理？

完成後執行 `test_linear_quantize()`，對照原始、量化、反量化三張矩陣。挑一個正數與一個負數手算，確認與圖中相符。

若失敗，依序檢查運算順序、rounding、dtype、zero point 符號。現有 helper 的窄整數轉型發生在最終 clamp 前；若額外嘗試極端輸入，也要留意中間值是否已溢位，不要直接更動給定流程。

## 4. Q2：由資料範圍決定 scale 與 zero point

位置：`get_quantization_scale_and_zero_point()`。

把實數最小值與最大值分別對應到整數兩個端點，寫成兩條方程式。先消去 zero point 求 scale，再代回其中一條方程式。先完成推導，再翻成 tensor 的極值與 Python scalar 運算。

檢查重點：

- 比較的是兩端點的「距離」，不要把可表示數值的個數當成距離。
- zero point 最後必須是合法範圍內的整數；題目已提供這部分處理。
- 回傳型別要符合 docstring：scale 是 Python `float`，zero point 是 Python `int`。

觀察後續 feature quantization 的結果，檢查實數 0 的對應與端點重建誤差。另想想全相同輸入會讓推導遇到什麼問題；先記錄這個邊界情況，勿自行修改禁止修改的 helper。

## 5. Q3：對稱權重量化與 per-channel scale

位置：`get_quantization_scale_for_weight()`；搭配閱讀已提供的 `linear_quantize_weight_per_channel()`。

這一題固定權重 zero point 為 0。先問自己：當最大絕對值要對應到正的整數上界時，scale 應如何決定？不要直接沿用 Q2 的非對稱範圍推導。

追蹤 helper 的形狀：

| 權重 | output channel 軸 | 對應的 scale shape |
| --- | --- | --- |
| Linear：`[oc, ic]` | 第 0 軸 | `[oc, 1]` |
| Conv2d：`[oc, ic, kh, kw]` | 第 0 軸 | `[oc, 1, 1, 1]` |

讀懂 `5e-7` 下限的用途：若整個 channel 都是 0，接下來的除法會怎樣？

完成後執行 notebook 的權重分布觀察，保留浮點、4-bit、2-bit 圖。比較同一層的離散程度與零值數量。這裡的 histogram 不能證明分類準確率，也不能證明模型檔案真的按位元數縮小。

## 6. Q4：bias 的量化尺度

位置：`linear_quantize_bias_per_output_channel()`。

從一個乘加運算開始：先寫出輸入與權重各自的實數表示，觀察兩者相乘後的尺度。bias 要加入這個累加值，因此必須處於相同尺度；再對照 HW.md 給出的 bias 條件。

檢查 bias 是否為每個 output channel 一個值、scale 是否也對應同一個 channel，以及回傳的量化 bias 是否為 `int32`。

目前 `test_quantized_fc()` 內有一行把計算結果與自己比較：`_quantized_bias.equal(_quantized_bias)`。它無法驗證 bias 的數值正確性。保留原測試，在允許的獨立驗證位置另外核對 bias 與題目給定的預期值；也要執行 Conv2d 測試，不能只憑 FC 的 passed 訊息下結論。

## 7. Q5：量化 Linear 的輸出還原與再量化

位置：`quantized_linear()`；先讀 `shift_quantized_linear_bias()`。

題目已完成乘加與最後的 rounding、clamp、dtype 轉換。你要處理的是：累加結果如何換到輸出量化的尺度，以及輸出 zero point 的位移。

先畫出這條資料流：

`輸入與權重的量化整數 → 累加值 → 輸出尺度下的數值 → 輸出量化整數`

思考：

- 累加值目前帶有哪兩個 scale？目的端使用哪個 scale？
- shifted bias 已吸收輸入 zero point 的哪一項？若再補償一次，會造成什麼錯誤？
- 輸出是 `[batch, oc]`，scale 應如何對齊 `oc`？以實際 tensor shape 為準，勿只照抄註解中的維度。

執行 `test_quantized_fc()`。失敗時先比 shape，再比縮放與位移，最後檢查 rounding。

## 8. Q6：量化 Conv2d

位置：`quantized_conv2d()`；先讀 `shift_quantized_conv2d_bias()`。

沿用 Q5 的尺度推理，但重新處理輸出 `[batch, oc, height, width]` 的 broadcasting。每個 channel 的 scale 應在所有 batch 與空間位置共用。

讀懂既有 padding：實數 0 對應量化域中的哪個值？為什麼目前使用 `input_zero_point` 填補，而不是直接填整數 0？

執行 `test_quantized_conv()`，對照通道圖與預期輸出。若只有部分通道錯誤，先檢查 scale 與 bias 的軸；若邊界位置錯誤，回頭理解 padding 與 zero point 的關係。

## 9. Q7：group-wise 權重量化

位置：`linear_quantize_weight_per_group()`。

先把一列權重畫成數個連續群組，標出每組哪些值共用 scale。依題目 docstring 列出分組後的 shape、求極值時應消去的軸，以及保留單一維度的理由。

完成後執行 `test_group_quantize()`，檢查：

- 量化權重的回傳 shape 是否仍為 `[oc, ic]`。
- scale 是否每一組各有一個，且可與該組 broadcasting。
- 全零群組是否有正的 scale。
- group size 變小後，誤差與 scale 數量如何改變。

別只寫「群組越小越好」：Part 2 還必須計入更多 scale 帶來的儲存成本。

## 10. Q8：以 MSE 比較 clipping 範圍

位置：`search_clipping_range()`。

先畫出每個候選值都要經過的流程：

`候選 clipping ratio → 範圍與 scale → 量化 → 反量化 → 與原始 tensor 比較 MSE`

關鍵問題：誤差應與原始資料比較，還是與已裁切的資料比較？若使用後者，會漏掉哪一種誤差？

執行 `test_search_clipping_range()`，檢查 MSE 列表順序是否與 `ratios` 一致、元素是否為 Python `float`。保留 MSE 曲線與分布圖，解釋縮小步距和裁切大值之間的取捨；不要直接把測試指定的最佳 ratio 寫死在函式中。

## 11. Q9：BatchNorm folding

位置：`fold_conv_bn()`。

先在紙上把 `Conv → BatchNorm` 的推論公式展開，分開收集輸入的係數與常數項，再對照 notebook 的公式。辨認 running mean、running variance、gamma、beta、eps 各自在哪裡出現。

檢查：

- 使用推論時的固定統計量，而非目前 batch 的統計量。
- 縮放因子沿 output channel 套用到權重。
- 有 bias 與沒有 bias 的情況都成立。
- 比較輸出時使用容許誤差，不要求浮點逐位相等。

執行 `test_fold_conv_bn()`，保留兩種 bias 設定的最大絕對誤差與前後輸出圖。再思考：為什麼量化時應觀察 folding 後的權重範圍？

## 12. Q10：fake quantization 與 STE

位置：`FakeQuantizeSTE.forward()`、`backward()`。

把 forward 與 backward 分開理解：forward 模擬量化後的數值，仍回傳浮點 tensor；backward 則依題目指定的 STE 規則傳遞梯度。

動手前回答：

- 為什麼直接經過 `round()` 的梯度不能達到本題目的？
- 可表示實數範圍如何從整數端點、scale、zero point 得到？
- mask 應根據原始輸入，還是已被 clamp 的輸出建立？
- 上游梯度不是全 1 時，如何確認你保留了它的大小？
- `backward()` 為什麼必須回傳四個位置？

執行 `test_fake_quantize_ste()`，分別核對 forward 值、區間內梯度、區間外梯度與端點規則。Part 1 的 STE 是學習題目；Part 2 禁止 fine-tuning，不能因此將訓練流程搬過去。

## 13. 完成時保留的證據

HW.md 的 Part 1 提交項目是 notebook，沒有另列獨立報告。建議保留以下執行輸出，供檢查與後續報告使用：

| 內容 | 圖或數據 | 可以支持的觀察 |
| --- | --- | --- |
| Q1–Q2 | 原始／量化／反量化矩陣、scale、zero point | 映射與重建誤差 |
| Q3、Q7 | 不同 bitwidth、不同分組的圖與誤差 | 精度與量化粒度的關係 |
| Q4–Q6 | bias 核對、FC／Conv 測試結果 | 尺度與通道對齊正確性 |
| Q8 | clipping ratio–MSE 曲線、最佳候選值 | 裁切範圍的取捨 |
| Q9 | 有／無 bias 的最大絕對誤差、輸出差異圖 | folding 的數值一致性 |
| Q10 | forward 階梯圖與梯度圖 | fake quantization 與 STE 行為 |

有數據才能確認結果（confirm），解釋時要交代推論依據（logic）。不從分布圖推論未量測的 accuracy 或推論速度。

最終重新執行整份 notebook，保留輸出，提交時依 HW.md 命名為 `Part1/StudentID_Part1.ipynb`。本地 `part1/` 是工作目錄，與繳交用的 `Part1/` 大小寫要分清楚。
