# 用 pi 數位序列驗證持續學習演算法：實驗報告

## 1. 目的

延續設計的持續學習（continual learning, CL）測試框架（accuracy matrix + 對角線可塑性探針 + BWT 遺忘指標），用 pi 的十進位小數位構造一個決定性、可重現、近乎不重複的長串流資料來源，實測七種演算法在 80 個依序到來的 task 上的表現：Naive（無防護下界）、EWC、Experience Replay、ReplayEWC、TaskBalancedReplay、Continual Backprop、ReplayContinualBP。

---

## 2. Benchmark 設計：Permuted-Pi-Digits

輸入是 pi 小數位序列上連續 K=8 位的滑動窗口，one-hot 編碼成 80 維向量。我們在實驗中發現了兩種截然不同的任務流設置，並針對它們進行了深入對照：

### 2.1 標籤隨機排列設置 (Label-Permuted Stream)
基礎分類規則為「計算窗口內 8 位數字總和，並依十分位分桶成 10 類」。對於每個任務 $t$，我們對這 10 類別使用一個任務特異的隨機排列（Permutation）重新編號標籤。
*   **單頭模型衝突 (數學極限約 10%)**：若使用傳統的單輸出頭 MLP，在沒有傳入 Task ID 的情況下，模型無法將同一個輸入 $X$ 同時映射到多個不同的隨機輸出值，其全域平均準確率理論上限為 $10\%$。
*   **多頭模型解決 (方案 A - Task-IL)**：為了解決此標籤衝突，我們為模型引入多頭（Multi-Head）結構。在訓練和測試任務 $t$ 時，僅使用與更新對應的第 $t$ 個輸出頭。

### 2.2 輸入隨機排列設置 (Input-Permuted / Domain-IL — 方案 B)
保持單個輸出頭不變，但將輸入特徵的 80 個維度依任務特異的 random permutation 進行重排（類似經典的 Permuted-MNIST）。所有任務共享完全一致的分類規則與同一個輸出映射。

---

## 3. 模型與演算法

兩層隱藏層 MLP（80→64→64→10，ReLU+softmax），純 numpy 手刻 forward/backward。程式現在包含七個 trainer，兩個模式的 80-task 完整表格皆已納入新增策略。

- **Naive**：純線上 SGD，無任何保護機制，作為下界基準。
- **EWC**：以 Fisher 資訊對角線錨定舊參數的二次懲罰項。方案 A 中二次懲罰主要應用於共享隱藏層參數。
- **Replay**：reservoir buffer（容量 500）。在方案 A 中，Replay 採用的重播樣本會根據其原本的 `task_idx` 通過對應的輸出頭計算梯度，並針對各輸出頭分別進行權重更新。
- **ReplayEWC**：在 Replay 的混合梯度上額外加入 online EWC 正則化。多頭模式下只保護共享隱藏層，讓 task head 保持可塑。
- **TaskBalancedReplay**：改用每個 task 各自的 reservoir，總 buffer 容量不變，但 slot 與抽樣都盡量平均分配到已看過的 task，避免長串流後早期任務樣本被全域 reservoir 稀釋。
- **Continual Backprop**：依效用（utility）選擇性重置低貢獻且夠老的隱藏單元，輸出端權重清零做函數保持式插入，其餘權重完全不動。多頭結構下，神經元重置時會對所有輸出頭的對應連線進行同步重置。
- **ReplayContinualBP**：把 TaskBalancedReplay 與 Continual Backprop 疊加，讓 replay 負責穩定性、神經元回收負責可塑性，是目前專案中最直接測試「記得住 + 學得動」互補性的候選方法。

實驗基本配置：lr=0.1、batch_size=10，7 個方法 × 3 個 seed（0/1/2），共計在 80 個任務（每個任務 4000 步）的長串流上跑滿。

---

## 4. 一個重要的 debug：EWC 數值爆炸

最初用 lam=50、Fisher 跨 task 無上限累加，在 n_tasks=10 的 smoke test 中於 task 2 訓練到一半時 weight norm 從 38 飆到 12,682 再到 inf/nan，準確率永久崩潰。這是因為二次懲罰項在做梯度下降時等價於一個彈簧系統，當某參數的 `lr × lam × fisher > 2` 時系統會幾何級數發散。
我們採用 online EWC（Schwarz et al. 2018）予以修正：Fisher 用衰減係數 γ=0.9 做指數移動累加而非無上限相加，lam 降到 20，並加上梯度範數裁剪作安全網。修正後在 lr=0.1、lam=20 下跑滿 80 個 task 沒有任何數值溢位警告。

---

## 5. 實驗結果對比

### 5.1 原始單頭標籤隨機排列（Single-Head, Label-Permuted）—— 標籤衝突 Baseline

| 方法 | 前 10 task 對角線均值 | 後 10 task 對角線均值 | BWT | 最終 80 task 平均準確率 | dead unit 比例（h2，前→後） | 有效秩（h1，前→後） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive | 58.9% | 13.6% | -20.4% $\pm$ 3.4% | **9.7%** $\pm$ 0.4% | 0.50 $\to$ 0.99 | 41.5 $\to$ 28.1 |
| EWC | 57.4% | 40.7% | -36.9% $\pm$ 1.3% | **10.4%** $\pm$ 0.7% | 0.34 $\to$ 0.75 | 43.1 $\to$ 35.4 |
| Replay | 44.3% | 55.2% | -41.3% $\pm$ 1.2% | **9.9%** $\pm$ 0.2% | 0.02 $\to$ 0.24 | 51.9 $\to$ 53.9 |
| ContinualBP | 61.6% | 56.5% | -49.7% $\pm$ 2.3% | **10.2%** $\pm$ 0.6% | 0.37 $\to$ 0.61 | 41.3 $\to$ 26.1 |

> [!CAUTION]
> 由於沒有任務特異的 Task ID 或獨立輸出頭，所有演算法在長串流結束時的平均準確率都塌縮至 **10% 左右**（純隨機猜測水準），數學上的標籤衝突使得全域穩定映射完全無法被學習。

---

### 5.2 方案 A：多頭結構 (Task-IL / `label_permuted`) 實驗結果

| 方法 | 前 10 task 對角線均值 | 後 10 task 對角線均值 | BWT | 最終 80 task 平均準確率 | dead unit 比例（h2，前→後） | 有效秩（h1，前→後） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive | 63.7% | 54.5% | -32.5% $\pm$ 0.7% | **29.1%** $\pm$ 6.1% | 0.34 $\to$ 0.90 | 41.8 $\to$ 23.2 |
| EWC | 64.3% | 78.3% | -16.8% $\pm$ 2.1% | **57.7%** $\pm$ 1.6% | 0.30 $\to$ 0.58 | 43.4 $\to$ 33.2 |
| Replay | 66.6% | 91.6% | -24.0% $\pm$ 1.2% | **60.5%** $\pm$ 1.1% | 0.08 $\to$ 0.39 | 47.7 $\to$ 34.6 |
| ReplayEWC | 65.8% | 88.5% | -22.2% $\pm$ 2.4% | **61.0%** $\pm$ 3.6% | 0.08 $\to$ 0.29 | 47.9 $\to$ 37.6 |
| TaskBalancedReplay | 68.8% | 87.4% | -29.5% $\pm$ 1.5% | **54.1%** $\pm$ 3.0% | 0.08 $\to$ 0.49 | 47.2 $\to$ 33.0 |
| ContinualBP | 62.3% | 69.7% | -38.3% $\pm$ 2.5% | **31.1%** $\pm$ 5.4% | 0.16 $\to$ 0.08 | 42.1 $\to$ 22.8 |
| ReplayContinualBP | 68.8% | 84.1% | -34.6% $\pm$ 2.5% | **48.1%** $\pm$ 2.3% | 0.01 $\to$ 0.00 | 47.5 $\to$ 34.8 |

> [!IMPORTANT]
> **多頭結構的突破**：改用多輸出頭後，**ReplayEWC**、**Replay** 與 **EWC** 的全域平均準確率分別達到 **61.0%**、**60.5%** 與 **57.7%**。這表明共享隱藏層成功維持了對 pi 數位求和的泛化表徵，且獨立的輸出頭能夠有效區分不同任務的排列標籤。ReplayEWC 在 final accuracy、BWT、mean forgetting 與 retention ratio 上都略優於原始 Replay，是目前這個專案裡最穩的持續學習候選方法。

#### 方案 A 實驗結果圖集
![方案 A - 80-task 可塑性曲線](fig1_diagonal_accuracy_label_permuted.png)
![方案 A - BWT 與最終平均準確率](fig2_bwt_finalacc_label_permuted.png)
![方案 A - 可塑性流失診斷 (有效秩/死神經元)](fig3_plasticity_diagnostics_label_permuted.png)

---

### 5.3 方案 B：輸入維度隨機排列 (Domain-IL / `input_permuted`) 實驗結果

| 方法 | 前 10 task 對角線均值 | 後 10 task 對角線均值 | BWT | 最終 80 task 平均準確率 | dead unit 比例（h2，前→後） | 有效秩（h1，前→後） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive | 52.7% | 25.5% | -30.4% $\pm$ 3.0% | **12.3%** $\pm$ 0.5% | 0.29 $\to$ 0.88 | 46.1 $\to$ 30.8 |
| EWC | 48.7% | 30.9% | -27.6% $\pm$ 0.3% | **11.0%** $\pm$ 0.3% | 0.20 $\to$ 0.56 | 47.7 $\to$ 46.3 |
| Replay | 29.1% | 22.0% | -11.5% $\pm$ 0.8% | **11.4%** $\pm$ 0.2% | 0.01 $\to$ 0.32 | 54.9 $\to$ 56.4 |
| ReplayEWC | 28.9% | 27.2% | -15.4% $\pm$ 0.1% | **11.3%** $\pm$ 0.3% | 0.01 $\to$ 0.22 | 54.9 $\to$ 56.5 |
| TaskBalancedReplay | 29.5% | 20.4% | -11.6% $\pm$ 1.0% | **11.4%** $\pm$ 0.5% | 0.01 $\to$ 0.29 | 54.7 $\to$ 56.3 |
| ContinualBP | 52.4% | 40.8% | -36.7% $\pm$ 1.0% | **10.9%** $\pm$ 0.1% | 0.18 $\to$ 0.27 | 47.3 $\to$ 45.6 |
| ReplayContinualBP | 29.6% | 18.0% | -11.9% $\pm$ 0.7% | **10.8%** $\pm$ 0.2% | 0.00 $\to$ 0.00 | 54.9 $\to$ 56.9 |

> [!NOTE]
> 在共享單輸出頭的前提下，輸入特徵維度的隨機排列使得模型難以跨越 80 個相異 domain。各演算法的最終平均準確率仍然大幅降至 **11% - 12%**（接近隨機猜測水準），遺忘依然十分嚴重。

#### 方案 B 實驗結果圖集
![方案 B - 80-task 可塑性曲線](fig1_diagonal_accuracy_input_permuted.png)
![方案 B - BWT 與最終平均準確率](fig2_bwt_finalacc_input_permuted.png)
![方案 B - 可塑性流失診斷 (有效秩/死神經元)](fig3_plasticity_diagnostics_input_permuted.png)

---

## 6. 深入討論與發現

### 6.1 可塑性與記憶的權衡（Stability-Plasticity Dilemma）
*   **ContinualBP** 在保護可塑性方面展現出極強的表現：在多頭設置下，其 h2 死神經元比例在 80 個任務後不升反降，維持在 **8%** (Naive 則惡化至 90%)；其 Late Diag Acc 仍高達 69.7%。這證實了神經元重置（utility resetting）能成功防止可塑性崩潰。然而，ContinualBP 的最終平均準確率（31.1%）低於 Replay 與 EWC，這是因為重置神經元旨在提供新的適應能力，但並不能防止已被重置單元上存儲的舊任務權重被覆盖（即缺乏舊知識保護機制）。
*   **Replay** 和 **EWC** 則專注於「穩定性」，通過回放或二次阻尼防止舊知識被篡改，在多頭 Task-IL 模式下分別取得了 60.5% 與 57.7% 的優異表現。
*   **ReplayEWC** 是目前最好的折衷：final average accuracy 達 61.0%，BWT 從 Replay 的 -24.0% 改善到 -22.2%，mean forgetting 從 26.0% 降到 24.7%，retention ratio 從 0.719 提升到 0.735。這表示「樣本級記憶 + shared-weight 保護」比單靠其中一者更接近持續學習的目標。
*   **ReplayContinualBP** 把 h2 dead unit 壓到 0%，但 final average accuracy 只有 48.1%，低於原始 Replay。這代表「維持可塑性」本身不是免費午餐：如果局部重置與回放更新沒有更細緻地協調，仍會破壞部分長期記憶。

### 6.2 BWT 遺忘指標被可塑性流失污染
如果只看 BWT 遺忘率，Naive 的 BWT 指標往往顯得「最不負」（看起來遺忘最少），這是一個嚴重的統計陷阱。
原因是 BWT 量的是最後準確率與剛訓練完準確率（對角線）的落差，而 Naive 的對角線準確率早就因為可塑性流失而崩潰。由於「剛學完就沒學好」，後期測試才顯得「沒忘掉多少」。因此，**對角線準確率與 BWT 遺忘指標必須同時解讀**，單獨觀察 BWT 會被可塑性流失嚴重干擾。

---

## 7. 本次改進：更接近「持續學習」的實驗閉環

這版程式把「持續學習效果」拆成三個可觀察面向，而不是只看單一 final accuracy：

1. **學得動**：對角線準確率 `A[t,t]`、late diagonal mean、plasticity slope。
2. **記得住**：final average accuracy、retention ratio（最後平均準確率 / 剛學完平均準確率）。
3. **忘多少**：BWT 與 mean forgetting（每個舊 task 歷史最佳表現到最後表現的落差）。

工程上新增兩個候選策略：

- **ReplayEWC** 把 Experience Replay 與 online EWC 疊加，是目前表現最好的穩定性增強方法。
- **TaskBalancedReplay** 修正全域 reservoir 在長任務流中的早期 task 稀釋問題。
- **ReplayContinualBP** 把 replay 的穩定性與 ContinualBP 的可塑性維護接在一起，對應本研究一開始提出的核心假設：單一機制通常只能解遺忘或可塑性流失其中一側，複合機制才有機會真正提升持續學習。

`analyze.py` 也改成會自動偵測結果檔裡的方法清單，並在沒有 matplotlib 的 Python 環境中仍可輸出 `summary_stats_*.json`，只跳過圖表產生。

## 8. 結論

1.  **資料流設計**：pi 數位序列能為持續學習提供可重現、非重複的數據流，但「預測下一位」本質不可學，必須改用「窗口求和分桶 + 標籤隨機排列」。
2.  **標籤衝突之解決**：在 80 個任務的超長標籤重映射下，必須採用多頭結構（Task-IL）方能打破單輸出頭帶來的數學矛盾，使 ReplayEWC、Experience Replay 與 EWC 的全域平均準確率顯著攀升至 50% 以上。
3.  **機制互補性**：ContinualBP 強於可塑性維護，EWC 與 Replay 強於舊記憶維持。ReplayEWC 顯示樣本級 replay 與 shared-weight EWC 可以互補；ReplayContinualBP 則提醒我們，神經元重置若沒有更細緻的保護機制，仍可能傷害長期記憶。

---

## 檔案說明

- `pi_digits.py`：產生/快取 pi 小數位序列。
- `benchmark.py`：Permuted-Pi-Digits 串流（支持多頭 `label_permuted` 和單頭 `input_permuted`）。
- `model.py`：支持多頭選擇與可塑性診斷的 numpy MLP 實現。
- `trainers.py`：適配多輸出頭、樣本重播分組、ReplayEWC、task-balanced replay 與神經元重置的七種 CL Trainer。
- `run.py` / `run_one_combo.py`：主實驗腳本（支持命令行參數選擇模式、方法、seed、任務數與輸出路徑）。
- `analyze.py`：彙整多 seed 實驗結果，輸出 JSON 與畫圖；缺 matplotlib 時仍會輸出 summary JSON。
- `results_label_permuted.json` / `results_input_permuted.json`：實驗原始數據。
- `summary_stats_label_permuted.json` / `summary_stats_input_permuted.json`：跨 seeds 彙整後數據。
- `fig1_diagonal_accuracy_*.png` / `fig2_bwt_finalacc_*.png` / `fig3_plasticity_diagnostics_*.png`：性能對比與診斷圖表。
