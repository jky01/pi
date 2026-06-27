# 用 pi 數位序列驗證持續學習演算法：實驗報告

## 0. 主要結果摘要（TL;DR）

純 numpy 手刻的 80→64→64→10 MLP，在 pi 數位構造的 Permuted-Pi-Digits 長串流上測 15 種持續學習機制。三個分階段的主要結論：

**(A) 兩種任務流、兩種瓶頸**（80 tasks，final average accuracy，3 seeds）

| 設置 | 最佳法 | 最佳 final | 離線上界 (Joint) | 瓶頸性質 |
| :--- | :--- | :---: | :---: | :--- |
| `label_permuted`（多頭 Task-IL） | HippocampalReplayEWC | **86.6%** | **99.3%** | 遺忘（還有 ~13% 空間） |
| `input_permuted`（單頭 Domain-IL） | — | 12% | **12%** | 結構性：單一共享輸入層無法反排列 80 個輸入 |

**(B) 輸入轉接器把 input_permuted 從「不可能」變成「可解」**（§7）

`input_permuted` 連離線上界都只有 12% → 證明不是遺忘問題。給每個 task 一個 identity 初始化的輸入 adapter 後：Naive 12%→20%（恢復可學性）、**ReplayEWC 12%→54.7%**、上界 12%→81.1%。所有 replay/記憶路徑都已接好 adapter。

**(C) 可永續學習：瓶頸是遺忘、不是可塑性流失**（§9，壓力測到 250 tasks）

- 只要有 replay，對角線（可塑性）一路上升不崩潰（0.72→0.90 @250 tasks）→ 可塑性根本不綁定。
- `SustainableReplayEWC`（Fisher 保護回收）修好了 ReplayContinualBP 的記憶損傷，但在有 replay 時換不到準確率。
- `BennaFusi`（複雜突觸，冪律遺忘）驗證有效（forget 0.23→0.03），但被 Fisher-selective EWC 支配；其棲位在 **replay-free**：80-task 下把 Naive 29%→53%、逼近 EWC 58% 且不存樣本、不算 Fisher。
- **設計準則**：能存樣本就用 replay(+EWC)；不能存樣本才換複雜突觸；可塑性注入只在完全沒有 replay 時才有意義。

**(D) 直攻遺忘的最終演算法：Replay + Fisher-EWC + DER++ logit 蒸餾**（§10）

遺忘是函數空間干擾，不是權重位移。實作可切換的 `FunctionSpaceReplay`（Replay + DER++ 蒸餾 + GPM 投影）做消融：GPM 因本 benchmark 輸入平穩而凍結共享層（不適配），但 **DER++ logit 蒸餾**有效——在 130-task 串流上把 final 從 ReplayEWC 的 0.815 抬到 **0.892**、**retention→1.01（淨遺忘≈0）**、BWT→0、方差砍 4 倍。**優勢隨串流變長而複利放大**：250 tasks 時領先從 +7.7 拉到 **+14.2 分**（0.680→0.822），retention 仍 0.97。

**(E) 校正＋補強：機制隨設定而變**（§11）。拿掉 task ID 改測 **Class-IL**（單頭、200 類）後，Task-IL 的英雄 **DER++ 反轉成有害**（病灶＝線性頭 recency bias）。但把讀出換成 **NCM 原型分類器（`NCMReplayEWC`，iCaRL 式）**就把缺口幾乎補滿：**0.312 → 0.858、遺忘 0.50→0.06、retention >1.0**（超過線性頭 Joint 0.742）。**誠實答案**：Task-IL 與 Class-IL 在本 benchmark 都已有接近上界的配置；跨設定穩健的骨幹是 **replay + Fisher-EWC**，再依設定換對的讀出（Task-IL：head+DER++；Class-IL：無偏原型）。距「通用持續學習」仍有任務衝突 / 無 buffer / 無邊界 / 規模等硬牆（見 `NEXT_STEPS.md`）。

---

## 1. 目的

延續設計的持續學習（continual learning, CL）測試框架（accuracy matrix + 對角線可塑性探針 + BWT 遺忘指標），用 pi 的十進位小數位構造一個決定性、可重現、近乎不重複的長串流資料來源，實測九種主表演算法在 80 個依序到來的 task 上的表現：Naive（無防護下界）、EWC、Experience Replay、ReplayEWC、SurpriseReplayEWC、HippocampalReplayEWC、TaskBalancedReplay、Continual Backprop、ReplayContinualBP。

### 1.1 持續學習的定義

持續學習不是單純「一直訓練」模型，而是指模型在資料或任務依序到來時，能夠持續吸收新知識，同時盡量保留舊知識的能力。更精準地說，持續學習要求模型在不重新從零訓練、也不一次看到所有歷史資料的情況下，依序學習新任務，並維持舊任務表現。

在本專案中，持續學習效果被拆成三個核心面向：

1. **學得動**：新任務到來時，模型仍能快速學會，而不是因為訓練太久、表徵僵化而失去可塑性。
2. **記得住**：學新任務後，不要把舊任務能力大幅覆蓋掉，也就是避免 catastrophic forgetting。
3. **可累積**：模型不是只在任務之間切換，而是能讓過去學到的表示、規則或經驗幫助未來學習。

因此，本報告不只看最後準確率，也同時觀察 `A[t,t]`（剛學完第 t 個任務時的準確率，代表可塑性）、`final_avg_acc`（全部任務學完後的平均保留能力）、`BWT`（學新任務對舊任務的影響）、`mean_forgetting`（歷史最佳表現到最後表現的退步量）與 `retention_ratio`（最後保留多少剛學會時的能力）。

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

兩層隱藏層 MLP（80→64→64→10，ReLU+softmax），純 numpy 手刻 forward/backward。程式現在包含十二個 trainer（含 §7 新增的 Joint 離線上界），其中九個已納入兩個模式的 80-task 完整表格；`DarkReplayEWC` 與 `MarginSurpriseReplayEWC` 是文獻啟發的實驗方法，暫不列入主表。

- **Naive**：純線上 SGD，無任何保護機制，作為下界基準。
- **EWC**：以 Fisher 資訊對角線錨定舊參數的二次懲罰項。方案 A 中二次懲罰主要應用於共享隱藏層參數。
- **Replay**：reservoir buffer（容量 500）。在方案 A 中，Replay 採用的重播樣本會根據其原本的 `task_idx` 通過對應的輸出頭計算梯度，並針對各輸出頭分別進行權重更新。
- **ReplayEWC**：在 Replay 的混合梯度上額外加入 online EWC 正則化。多頭模式下只保護共享隱藏層，讓 task head 保持可塑。調參後預設 buffer capacity 從 500 提升到 2000。
- **SurpriseReplayEWC**：在 ReplayEWC 上加入 surprise-prioritized sampling；每步先從 buffer 抽候選池，再回放目前模型 cross-entropy loss 最高的舊樣本。
- **HippocampalReplayEWC**：在人腦「海馬迴快速情節記憶 + 皮質慢速參數學習」的啟發下，沿用 SurpriseReplayEWC 作為慢速 learner，並在推論時從 replay buffer 建立 task/context episodic prototypes。記憶讀出採 uncertainty-gated blending：MLP 越不確定，episodic memory 權重越高；MLP 已有把握時，記憶庫少干預。
- **MarginSurpriseReplayEWC**：在 SurpriseReplayEWC 上加入 top-2 margin 訊號；高 loss 捕捉已忘掉的舊樣本，低 margin 捕捉舊任務決策邊界附近的脆弱樣本，對應 episodic memory / adaptive replay 文獻中「保留最能保護舊行為的 exemplar」這條路線。
- **DarkReplayEWC**：實驗性方法，在 ReplayEWC 上保存舊 logits 並做 logit consistency replay；短流 sweep 中沒有贏 ReplayEWC，因此不列入主結果表。
- **TaskBalancedReplay**：改用每個 task 各自的 reservoir，總 buffer 容量不變，但 slot 與抽樣都盡量平均分配到已看過的 task，避免長串流後早期任務樣本被全域 reservoir 稀釋。
- **Continual Backprop**：依效用（utility）選擇性重置低貢獻且夠老的隱藏單元，輸出端權重清零做函數保持式插入，其餘權重完全不動。多頭結構下，神經元重置時會對所有輸出頭的對應連線進行同步重置。
- **ReplayContinualBP**：把 TaskBalancedReplay 與 Continual Backprop 疊加，讓 replay 負責穩定性、神經元回收負責可塑性，是目前專案中最直接測試「記得住 + 學得動」互補性的候選方法。

實驗基本配置：lr=0.1、batch_size=10，9 個主方法 × 3 個 seed（0/1/2），共計在 80 個任務（每個任務 4000 步）的長串流上跑滿。

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
| ReplayEWC | 68.8% | 90.7% | -3.5% $\pm$ 2.4% | **81.8%** $\pm$ 3.7% | 0.12 $\to$ 0.37 | 45.7 $\to$ 34.9 |
| SurpriseReplayEWC | 76.2% | 93.4% | -2.5% $\pm$ 2.5% | **85.8%** $\pm$ 3.0% | 0.15 $\to$ 0.30 | 44.3 $\to$ 38.2 |
| HippocampalReplayEWC | 82.0% | 93.5% | -4.2% $\pm$ 2.2% | **86.6%** $\pm$ 2.7% | 0.15 $\to$ 0.30 | 44.3 $\to$ 38.2 |
| TaskBalancedReplay | 68.8% | 87.4% | -29.5% $\pm$ 1.5% | **54.1%** $\pm$ 3.0% | 0.08 $\to$ 0.49 | 47.2 $\to$ 33.0 |
| ContinualBP | 62.3% | 69.7% | -38.3% $\pm$ 2.5% | **31.1%** $\pm$ 5.4% | 0.16 $\to$ 0.08 | 42.1 $\to$ 22.8 |
| ReplayContinualBP | 68.8% | 84.1% | -34.6% $\pm$ 2.5% | **48.1%** $\pm$ 2.3% | 0.01 $\to$ 0.00 | 47.5 $\to$ 34.8 |

> [!IMPORTANT]
> **多頭結構的突破**：改用多輸出頭後，**HippocampalReplayEWC**、**SurpriseReplayEWC**、**ReplayEWC**、**Replay** 與 **EWC** 的全域平均準確率分別達到 **86.6%**、**85.8%**、**81.8%**、**60.5%** 與 **57.7%**。這表明共享隱藏層成功維持了對 pi 數位求和的泛化表徵，且獨立的輸出頭能夠有效區分不同任務的排列標籤。HippocampalReplayEWC 在 final average accuracy 上小幅超過 SurpriseReplayEWC，但 BWT 略差，因此更像是「提高最終可用表現」而非全面支配所有遺忘指標。

#### 方案 A 實驗結果圖集
![方案 A - 80-task 可塑性曲線，準確率越高越好](fig1_diagonal_accuracy_label_permuted.png)
![方案 A - BWT 與最終平均準確率，BWT 越接近 0 越好，最終準確率越高越好](fig2_bwt_finalacc_label_permuted.png)
![方案 A - 可塑性流失診斷，dead unit 越低越好，有效秩越高越好](fig3_plasticity_diagnostics_label_permuted.png)

---

### 5.3 方案 B：輸入維度隨機排列 (Domain-IL / `input_permuted`) 實驗結果

| 方法 | 前 10 task 對角線均值 | 後 10 task 對角線均值 | BWT | 最終 80 task 平均準確率 | dead unit 比例（h2，前→後） | 有效秩（h1，前→後） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive | 52.7% | 25.5% | -30.4% $\pm$ 3.0% | **12.3%** $\pm$ 0.5% | 0.29 $\to$ 0.88 | 46.1 $\to$ 30.8 |
| EWC | 48.7% | 30.9% | -27.6% $\pm$ 0.3% | **11.0%** $\pm$ 0.3% | 0.20 $\to$ 0.56 | 47.7 $\to$ 46.3 |
| Replay | 29.1% | 22.0% | -11.5% $\pm$ 0.8% | **11.4%** $\pm$ 0.2% | 0.01 $\to$ 0.32 | 54.9 $\to$ 56.4 |
| ReplayEWC | 30.2% | 17.4% | -11.6% $\pm$ 0.1% | **10.6%** $\pm$ 0.4% | 0.01 $\to$ 0.00 | 53.9 $\to$ 57.5 |
| SurpriseReplayEWC | 20.7% | 17.0% | -5.9% $\pm$ 1.1% | **11.0%** $\pm$ 0.1% | 0.05 $\to$ 0.03 | 53.2 $\to$ 56.3 |
| HippocampalReplayEWC | 21.6% | 16.6% | -5.4% $\pm$ 1.4% | **10.6%** $\pm$ 0.2% | 0.05 $\to$ 0.03 | 53.2 $\to$ 56.3 |
| TaskBalancedReplay | 29.5% | 20.4% | -11.6% $\pm$ 1.0% | **11.4%** $\pm$ 0.5% | 0.01 $\to$ 0.29 | 54.7 $\to$ 56.3 |
| ContinualBP | 52.4% | 40.8% | -36.7% $\pm$ 1.0% | **10.9%** $\pm$ 0.1% | 0.18 $\to$ 0.27 | 47.3 $\to$ 45.6 |
| ReplayContinualBP | 29.6% | 18.0% | -11.9% $\pm$ 0.7% | **10.8%** $\pm$ 0.2% | 0.00 $\to$ 0.00 | 54.9 $\to$ 56.9 |

> [!NOTE]
> 在共享單輸出頭的前提下，輸入特徵維度的隨機排列使得模型難以跨越 80 個相異 domain。各演算法的最終平均準確率仍然大幅降至 **11% - 12%**（接近隨機猜測水準），遺忘依然十分嚴重。

#### 方案 B 實驗結果圖集
![方案 B - 80-task 可塑性曲線，準確率越高越好](fig1_diagonal_accuracy_input_permuted.png)
![方案 B - BWT 與最終平均準確率，BWT 越接近 0 越好，最終準確率越高越好](fig2_bwt_finalacc_input_permuted.png)
![方案 B - 可塑性流失診斷，dead unit 越低越好，有效秩越高越好](fig3_plasticity_diagnostics_input_permuted.png)

---

## 6. 深入討論與發現

### 6.1 可塑性與記憶的權衡（Stability-Plasticity Dilemma）
*   **ContinualBP** 在保護可塑性方面展現出極強的表現：在多頭設置下，其 h2 死神經元比例在 80 個任務後不升反降，維持在 **8%** (Naive 則惡化至 90%)；其 Late Diag Acc 仍高達 69.7%。這證實了神經元重置（utility resetting）能成功防止可塑性崩潰。然而，ContinualBP 的最終平均準確率（31.1%）低於 Replay 與 EWC，這是因為重置神經元旨在提供新的適應能力，但並不能防止已被重置單元上存儲的舊任務權重被覆盖（即缺乏舊知識保護機制）。
*   **Replay** 和 **EWC** 則專注於「穩定性」，通過回放或二次阻尼防止舊知識被篡改，在多頭 Task-IL 模式下分別取得了 60.5% 與 57.7% 的優異表現。
*   **ReplayEWC** 是目前最好的折衷：final average accuracy 達 81.8%，BWT 從 Replay 的 -24.0% 改善到 -3.5%，mean forgetting 從 26.0% 降到 11.1%，retention ratio 從 0.719 提升到 0.959。這表示「足夠舊樣本覆蓋率 + shared-weight 保護」比單靠其中一者更接近持續學習的目標。
*   **SurpriseReplayEWC** 進一步把 final average accuracy 提升到 85.8%，mean forgetting 降到 8.9%，retention ratio 提升到 0.971。這表示在 buffer 已經足夠大時，「回放哪些樣本」開始變得比單純增加容量更關鍵。
*   **HippocampalReplayEWC** 把 final average accuracy 再小幅推到 86.6%，early diagonal mean 也從 SurpriseReplayEWC 的 76.2% 提升到 82.0%。這符合「快速情節記憶可補足慢速參數學習」的直覺。不過它的 BWT 為 -4.2%，略差於 SurpriseReplayEWC 的 -2.5%，表示 episodic readout 改善了最終可用表現，但不代表 shared parameters 本身更少被改動。
*   **ReplayContinualBP** 把 h2 dead unit 壓到 0%，但 final average accuracy 只有 48.1%，低於原始 Replay。這代表「維持可塑性」本身不是免費午餐：如果局部重置與回放更新沒有更細緻地協調，仍會破壞部分長期記憶。

### 6.2 BWT 遺忘指標被可塑性流失污染
如果只看 BWT 遺忘率，Naive 的 BWT 指標往往顯得「最不負」（看起來遺忘最少），這是一個嚴重的統計陷阱。
原因是 BWT 量的是最後準確率與剛訓練完準確率（對角線）的落差，而 Naive 的對角線準確率早就因為可塑性流失而崩潰。由於「剛學完就沒學好」，後期測試才顯得「沒忘掉多少」。因此，**對角線準確率與 BWT 遺忘指標必須同時解讀**，單獨觀察 BWT 會被可塑性流失嚴重干擾。

---

## 7. 上界基準線與輸入轉接器（本輪新增）

先前只有下界（Naive）與強參考（Replay/EWC），缺一條真正的上界，導致「86.6% 算好還是不好」沒有定義。本輪補上離線多任務上界 **Joint**，並針對 `input_permuted` 一直破不了的瓶頸做了一個結構性介入：**per-task input adapter（輸入轉接器）**。

### 7.1 Joint 離線上界：重新定義天花板

`JointTrainer` 儲存所有看過的樣本，每步從「目前為止所有 task 的聯集」均勻抽 mini-batch 做 i.i.d. 更新，等於拿掉持續學習的循序限制。它蓄意打破計算/記憶體預算公平性，只當天花板基準，且因為它不是循序學習者，**只有 `final_avg_acc` 有意義，其 BWT/對角線是假象**（會出現 BWT 為正、retention > 1 等數值）。

| 設置 | Joint final average accuracy（3 seeds） |
| :--- | :---: |
| `label_permuted`（多頭） | **99.3% ± 0.2%** |
| `input_permuted`（單頭，無 adapter） | **12.1% ± 0.2%** |

兩個結果都很關鍵：

1. **label_permuted 的真天花板是 99.3%，不是先前對角線暗示的 ~93%。** 目前最佳的 HippocampalReplayEWC（86.6%）其實還離上界有 **~13%** 的空間，先前「已接近極限」的判斷是錯的——是缺了上界才看不出來。
2. **input_permuted 的 12% 不是遺忘造成的。** 連「看過全部資料、完全不受遺忘限制」的離線上界都只有 12%（≈ 隨機猜測），證明這個瓶頸是**結構性、資訊理論層級**的：單一共享的第一層 `W1` 在數學上無法同時把 80 個不同的輸入排列 `P_t` 對應回同一個 `sum→bucket` 函數。這條過去被當成「持續學習失敗」的曲線，其實根本不是持續學習問題。

### 7.2 輸入轉接器：把「不可能」變成「一般的持續學習問題」

針對上述結構瓶頸，我們給每個 task 一個 identity 初始化的 `in_dim × in_dim` 線性轉接器 `A_t`，在進 `W1` 之前先做 `X @ A_t`。`A_t` 學會把該 task 的排列輸入轉回共享網路的正則空間，於是 `W1/W2/W3` 只需學一份共享的 `sum→bucket` 計算。轉接器是 task-specific 的（像輸出頭一樣），**不受 EWC 正則**。梯度以數值微分驗證為精確（float64 下相對誤差 ~1e-9）。

| 設置（`input_permuted`，80 tasks，3 seeds） | final average accuracy | BWT | mean forgetting | diag（early→late） |
| :--- | :---: | :---: | :---: | :---: |
| Naive（無 adapter） | 12.3% | -30.4% | 0.30 | — |
| Joint（無 adapter，上界） | 12.1% | — | — | — |
| **Naive + adapter** | **19.7% ± 4.0%** | -46.4% | 0.48 | 0.60→0.64 |
| SurpriseReplayEWC + adapter | **48.4% ± 0.8%** | -21.4% | 0.27 | 0.71→0.67 |
| HippocampalReplayEWC + adapter | **49.8% ± 0.7%** | -21.6% | 0.26 | 0.75→0.67 |
| **ReplayEWC + adapter** | **54.7% ± 1.2%** | -14.8% | 0.22 | 0.66→0.69 |
| **Joint + adapter（上界）** | **81.1% ± 0.5%** | — | — | — |

![input_permuted：輸入轉接器打破結構性下限（final average accuracy，越高越好）](fig4_input_permuted_adapter_ladder.png)

三個重點：

1. **轉接器恢復了「可學性」，但沒解決遺忘。** 加上轉接器後，Naive 的對角線可塑性立刻回到 0.60（無 adapter 時 diag_first 只有 ~0.23），代表每個 task 變得學得動了；但 Naive 仍災難性遺忘（BWT -46%），final 只有 19.7%。也就是說，**轉接器把 input_permuted 從「結構上不可學（連 Joint 都只有 12%）」轉換成「一般的穩定性–可塑性問題（可學、但會忘）」**——而後者正是 replay/EWC 擅長的場景。

2. **接好 adapter 的 replay 把 12% 抬到 55%。** 為了讓 replay 在 adapter 下正確運作，我們把所有重播/記憶讀取路徑改成依 task 分組、各自套用對應 adapter（單頭共享輸出頭、但每個 task 走自己的 adapter）。修好之後 **ReplayEWC + adapter 達到 54.7%**，BWT 從 Naive 的 -46% 改善到 -15%，回收了轉接器打開的 19.7% → 81.1% 空間的一半以上。

3. **方法排名翻轉。** 在 `label_permuted`：Hippocampal > Surprise > ReplayEWC；在 `input_permuted + adapter`：**ReplayEWC > Hippocampal > Surprise**。在這個 regime，surprise-prioritized sampling 與 episodic memory 反而拖後腿（遺忘 0.27 vs ReplayEWC 0.22）——很可能是因為轉接器還在學的過程中，「surprise」會過度偏向 adapter 尚未收斂的樣本，而 episodic prototype 所在的特徵空間也還在漂移。在這裡**樸素的 replay+EWC 反而最穩健**。剩下 55% → 81% 的差距主要是遺忘（diag_last ~0.69，可塑性沒問題），下一步的槓桿是更大的 buffer / 更強的 shared-layer EWC，而不是更花俏的取樣。

---

## 8. 本次改進：更接近「持續學習」的實驗閉環

這版程式把「持續學習效果」拆成三個可觀察面向，而不是只看單一 final accuracy：

1. **學得動**：對角線準確率 `A[t,t]`、late diagonal mean、plasticity slope。
2. **記得住**：final average accuracy、retention ratio（最後平均準確率 / 剛學完平均準確率）。
3. **忘多少**：BWT 與 mean forgetting（每個舊 task 歷史最佳表現到最後表現的落差）。

最新文獻對應到三個可落地方向：DER/DER++ 與 SER 類方法強調 logits consistency；IDER 類方法把 replay consistency 做成可疊加框架；2024-2025 的 adaptive/scalable replay 工作則強調記憶容量、回放效率與樣本選擇。這次對應實作如下：

- **ReplayEWC** 把 Experience Replay 與 online EWC 疊加，是目前表現最好的穩定性增強方法。
- **SurpriseReplayEWC** 把 sample selection 加入 ReplayEWC，優先回放目前模型最意外、loss 最高的舊樣本，是目前最佳的純參數更新/重播 baseline。
- **HippocampalReplayEWC** 加入 uncertainty-gated episodic prototype memory，對應「海馬迴快速情節記憶 + 皮質慢速學習」的腦啟發假設。完整 80-task 多頭實驗中，final average accuracy 從 SurpriseReplayEWC 的 85.8% 小幅提升到 86.6%。
- **MarginSurpriseReplayEWC** 延伸 SurpriseReplayEWC，額外偏好 top-2 margin 小的舊樣本，讓 replay 更聚焦於舊任務決策邊界。這是把 Google DeepMind episodic-memory 方向與 adaptive replay/prioritized replay 想法放進本專案的小型可跑版本。
- **DarkReplayEWC** 實作 logits consistency replay，呼應 DER/SER/IDER 類文獻，但在本 benchmark 的短流 sweep 中沒有勝過 ReplayEWC，暫保留為實驗 baseline。
- **TaskBalancedReplay** 修正全域 reservoir 在長任務流中的早期 task 稀釋問題。
- **ReplayContinualBP** 把 replay 的穩定性與 ContinualBP 的可塑性維護接在一起，對應本研究一開始提出的核心假設：單一機制通常只能解遺忘或可塑性流失其中一側，複合機制才有機會真正提升持續學習。

新增的 `MarginSurpriseReplayEWC` 已用 `margin_weight=1.0` 跑過完整 80-task 驗證。它在 `label_permuted` 模式得到 final average accuracy **84.0% ± 2.9%**、BWT **-4.0% ± 1.5%**、mean forgetting **9.8% ± 1.4%**，沒有超過 `SurpriseReplayEWC`（85.8% ± 3.0%）。在 `input_permuted` 模式則仍停在 **10.8% ± 0.2%**，表示低 margin replay 無法解決單頭 domain-IL 的根本瓶頸。因此它目前保留為「新論文方向的實驗候選」，暫不升級為主表預設方法。

文獻對應：`SurpriseReplayEWC` 對應 SuRe 的 surprise-prioritized replay；`DarkReplayEWC` 對應 Dark Experience Replay 的 logits consistency；`HippocampalReplayEWC` 與 `MarginSurpriseReplayEWC` 對應 episodic memory / retrieval 可補足 parametric learning 的方向；Google DeepMind 的 aligned model merging 則比較適合下一步做 task-end consolidation，而不是直接塞進目前的小型 MLP 主流程。

後續調參發現三個非常實用的結果：短流 sweep 裡調大 `replay_batch` 看似有效，但完整 80-task 驗證反而退步；真正有效的是把 ReplayEWC 的 buffer capacity 從 500 提升到 2000。再往後，加入 surprise-prioritized sampling 又把 final average accuracy 從 81.8% 推到 85.8%。最後，固定比例的 episodic memory 在短流看似有利但長流會拖累；改成 uncertainty-gated episodic recall 後，HippocampalReplayEWC 才在完整 80-task 上小幅提升到 86.6%。這代表本 benchmark 的主要瓶頸先是舊任務樣本覆蓋不足，容量足夠後轉為回放樣本選擇，再往後則需要更細緻地決定「何時相信參數模型、何時召回記憶」。

`analyze.py` 也改成會自動偵測結果檔裡的方法清單，並在沒有 matplotlib 的 Python 環境中仍可輸出 `summary_stats_*.json`，只跳過圖表產生。

## 9. 可永續學習探針：可塑性流失 vs 遺忘，哪個才是長串流的瓶頸？

「可永續學習」要求模型在無上限的串流上同時不遺忘（失效 A）且不流失可塑性（失效 B）。為了找出本 benchmark 真正的瓶頸，我們把串流拉長並新增一個直接針對 §6.1 開放問題的方法。

### 9.1 新方法：`SustainableReplayEWC`（Fisher 保護的神經元回收）

§6.1 指出 ReplayContinualBP 的問題：神經元盲目重置會覆寫舊任務權重（48% < 純 Replay 60%）。`SustainableReplayEWC` 在 ReplayEWC 上加可塑性維護，但讓回收「對舊任務記憶有感」：(1) 只回收同時低效用且 **Fisher 重要度低**（對舊任務不關鍵）的成熟單元；(2) 被回收的單元連同其 EWC anchor/Fisher 一併清掉，避免正則項把新權重拉回死亡值。

### 9.2 長串流結果：可塑性在有 replay 時根本不綁定

| 130 tasks × 4000 steps（label_permuted，2 seeds） | final | retention | dead_h2 e→l | effrank_h1 e→l |
| :--- | :---: | :---: | :---: | :---: |
| ReplayEWC | **0.815** | **0.921** | 0.12→0.42 | 45.6→32.3 |
| ReplayContinualBP | 0.460 | 0.538 | 0.00→0.01 | 47.3→29.0 |
| **SustainableReplayEWC** | 0.706 | 0.804 | 0.02→**0.00** | 45.7→**35.3** |

把串流再拉到 **250 tasks × 2000 steps**，ReplayEWC 的對角線（可塑性）不但沒崩，反而**單調上升**：50-task 分箱為 `0.72 → 0.82 → 0.83 → 0.86 → 0.90`，即使它的 dead_h2 一路爬到 0.48。SustainableReplayEWC 把 dead_h2 壓在 ~0、effective rank 全場最高，對角線軌跡卻幾乎與 ReplayEWC 重合（0.73→0.87），final 反而較低（0.587 vs 0.680）且方差更大。`input_permuted + adapter` 也是同一個型態：對角線整段持平在 ~0.70，dead_h2 升到 0.38，但學習本身不受影響。

### 9.3 結論：瓶頸是遺忘，不是可塑性流失

三個關鍵推論：

1. **只要有 replay，多頭網路在 250 個任務內都不會可塑性崩潰**——對角線持續上升，dead-unit 比例升高並不轉化成學不動。亦即 dead-unit / effective rank 這些微觀徵兆在「有 replay」時不是學習的綁定限制。
2. **`SustainableReplayEWC` 完成了它的設計目標、但解了一個本 regime 不存在的問題。** 它是唯一同時做到 retention > 0.8、dead-unit ≈ 0、effective rank 最高的方法，也確實修好了 ReplayContinualBP 的記憶損傷（0.71/0.80 vs 0.46/0.54，證明 Fisher 保護有效）；但因為 ReplayEWC 本來就沒有可塑性瓶頸，回收買到的可塑性餘裕**換不到任何準確率**，反而引入擾動。
3. **本 benchmark 真正的可永續瓶頸是遺忘**：對角線（剛學完 ~0.70–0.90）與 final（~0.55–0.82）之間的差距全是遺忘。下一步提升「可永續性」的正確槓桿是壓低殘餘遺忘（更大/更聰明的記憶、更強的 shared-layer 保護、power-law 遺忘的 Benna-Fusi 複雜突觸、generative replay），而不是再加可塑性機制。可塑性注入要見效，得換到 **沒有 replay** 或 replay 嚴重不足的 regime。

### 9.4 Benna-Fusi 複雜突觸：冪律遺忘有效，但 isotropic consolidation 不敵 Fisher-selective EWC

依 §9.3 的指引，我們實作 Benna & Fusi (2016) 的複雜突觸（`BennaFusi` / `BennaFusiReplay`）：把每個共享權重換成一條 N 級、時間尺度幾何遞增的耦合變數鏈，梯度只打進最快的可見變數，鬆弛步驟把值往慢變數擴散；慢變數的共識會把可見權重往回拉，給出**冪律（而非指數）遺忘**，且完全不儲存樣本、不算 Fisher。

機制如預期：純 `BennaFusi` 把 20-task 的 mean forgetting 從 Naive 的 0.226 壓到 0.026、retention 從 0.67 拉到 >1.0，weight norm 受封閉鏈約束而穩定。`bf_dt` 是穩定↔可塑的旋鈕（越大越穩、越不可塑）。

但兩個對照給出清楚的定位：

| 80 tasks label_permuted（replay-free，mean of 2 seeds） | final | forget | retention |
| :--- | :---: | :---: | :---: |
| Naive | 0.291 | 0.342 | 0.470 |
| ContinualBP | 0.311 | 0.395 | 0.448 |
| **BennaFusi (dt=0.05)** | **0.533** | **0.122** | **0.884** |
| EWC | 0.577 | 0.212 | 0.777 |

1. **有 replay 時，Benna-Fusi 被 ReplayEWC 完全支配。** 40-task 下 `BennaFusiReplay`（無 EWC）即使把耦合調到極弱（dt=0.001），對角線仍只有 0.73（ReplayEWC 0.87），final 0.75 < 0.82。原因有原理性：EWC 是 **Fisher 選擇性**懲罰（只擋對舊任務重要的方向），Benna-Fusi 是 **各向同性**地拖住所有權重，於是用同樣的遺忘保護付出更多可塑性代價——在 replay 已覆蓋舊資料時，選擇性方法勝出。
2. **Benna-Fusi 的真正棲位是 replay-free / Fisher-free 線上鞏固。** 上表中它把 Naive 的 0.291 幾乎翻倍到 0.533、逼近 EWC 的 0.577，而且 forgetting（0.122）比 EWC（0.212）更低、retention 更高——關鍵是它**完全線上、不需要任何 per-task Fisher 計算、不存任何樣本**。這正是對話 §8 所說「多時間尺度突觸當記憶基質」的低成本實例。

結論：Benna-Fusi 驗證了冪律遺忘確實有效，但在本 benchmark「有 replay」的主線上，Fisher-selective EWC 是更有效率的權重穩定機制；Benna-Fusi 的價值在**無法存樣本、也不想算 Fisher 的純線上場景**。

## 10. 直攻遺忘：函數空間抗遺忘與混合式訓練器

§9 確立「瓶頸是遺忘」後，本節直接針對遺忘做演算法設計。核心理念（呼應對話 §11 的 functional regularization）：**遺忘是函數空間的干擾，不是權重空間的位移**——權重在舊任務的零空間裡大動也不會遺忘，在敏感方向小動卻會災難性遺忘。EWC 只是這件事的對角近似（忽略參數相關性），Benna-Fusi 更只是各向同性，所以都留有上界差距。我們實作 `FunctionSpaceReplay`，把三個直攻函數空間的機制疊在一起並可切換：

1. **Replay**（覆蓋）。
2. **DER++ logit 蒸餾**（函數軟錨）：回放時用 MSE 把現在對舊樣本的輸出拉回寫入當下的 logits，保住整個 softmax 幾何。
3. **GPM 梯度投影**（子空間硬鎖）：維護每個共享層「舊任務輸入子空間」的正交基 M，把共享層梯度投影到 M 的正交補（`G ← G − M(MᵀG)`），使舊輸入的輸出數學上不被擾動。

### 10.1 消融：GPM 不適配本 benchmark，但 DER++ 有效（40 tasks，label_permuted）

| 配置 | final | retention | forget | diag_last |
| :--- | :---: | :---: | :---: | :---: |
| ReplayEWC（baseline） | 0.818 | 1.016 | 0.102 | 0.873 |
| Replay only | 0.824 | 1.023 | 0.090 | 0.864 |
| +DER++（dark） | 0.804 | 1.042 | **0.047** | 0.844 |
| +GPM（dark+gpm） | **0.370** | 0.906 | 0.069 | **0.404** |

- **GPM 崩盤**（final 0.80→0.37、diag 0.84→0.40），這是有原理的負結果：GPM 防的是**輸入分布漂移**，但本 benchmark 的輸入是**平穩的**（label_permuted 漂的是標籤、由 head 處理；input_permuted 由 adapter 還原成正則輸入）。於是 task 0 之後 GPM 基就吃掉幾乎整個輸入子空間、**凍結共享層**→可塑性崩潰。函數空間的「原則」對，但 input-subspace 投影對平穩輸入是錯的工具。
- **DER++（logit 蒸餾）有效**：把 forgetting 從 0.090 砍到 0.047、retention 拉到 1.042。在 40 tasks 因為遺忘還沒累積，final 沒動；但這個遺忘削減會在長串流上複利放大。

### 10.2 在長串流上，DER++ 幾乎消滅淨遺忘（130 tasks，label_permuted，2 seeds）

把有效配置（Replay + Fisher-EWC + DER++ logit 蒸餾，α=0.5，GPM 關閉）＝ `DarkReplayEWC` 拉到 130 個任務：

| 130 tasks | final | forget | retention | BWT |
| :--- | :---: | :---: | :---: | :---: |
| ReplayEWC | 0.815 ± 0.074 | 0.130 | 0.921 | −0.07 |
| **DarkReplayEWC (+DER++)** | **0.892 ± 0.019** | **0.043** | **1.01** | **≈ 0** |

logit 蒸餾把 final 抬 **+7.7 分**、forgetting 砍到 **1/3**、retention 推到 **~1.0（淨遺忘幾乎為零）**、BWT→0，並把 seed 間方差砍掉 **4 倍**（0.074→0.019，更可靠）。報告早期短流 sweep（80 tasks、α=0.1）曾認為 DarkReplayEWC 沒贏 ReplayEWC——本節證明那是因為**遺忘未累積且蒸餾太弱**；長串流＋α=0.5 後，函數錨定的優勢清楚顯現。

再把串流拉到 **250 tasks × 2000 steps（3 seeds）**確認可永續性：

| 250 tasks | final | forget | retention | diag_last |
| :--- | :---: | :---: | :---: | :---: |
| ReplayEWC | 0.680 ± 0.051 | 0.206 | 0.823 | 0.912 |
| **DarkReplayEWC (+DER++)** | **0.822 ± 0.015** | **0.077** | **0.972** | 0.949 |

**優勢隨串流長度複利放大**：130 tasks 領先 +7.7 分、250 tasks 拉開到 **+14.2 分**；遺忘累積得越多，蒸餾擋掉的越多，而 diag_last 0.949 顯示可塑性毫髮無傷。retention 在 250 個任務後仍維持 **0.97**——這正是「可永續持續學習」的定義：抗遺忘能力**隨串流變長而變強**。

**這就是直接回答「找一個有效抗遺忘、達到持續學習的演算法」：Replay + Fisher-selective EWC + DER++ logit 蒸餾**。它在 130-task 串流上把淨遺忘壓到接近零（retention 1.01），是本專案目前最接近「持續學習」定義的配置。GPM 則被本 benchmark 的平穩輸入結構排除——這本身是一條清楚的設計教訓：input-subspace 投影只在輸入真的漂移時才該用。

### 10.3 正向遷移（可累積）：表徵層有、學習速度沒有

報告 §1.1 的第三個目標是「可累積（過去幫助未來）」。用現有資料檢驗（250 tasks）：

| | 串流早段 → 晚段 |
| :--- | :--- |
| 每任務最終準確率 A[t,t]（DER++） | 0.69 → 0.81 → 0.88 → 0.92 → **0.94** |
| 學習速度（first-batch acc） | 0.15 → 0.15 → 0.20 → 0.17 → 0.18（**持平**） |

- **表徵層正向遷移：有。** 同樣 2000 步預算下，晚段任務的最終準確率明顯更高（0.69→0.94）——累積的共享表徵確實幫了新任務；DER++ 因 retention 更好，累積得比 ReplayEWC 更乾淨（rise 0.69→0.94 vs 0.72→0.90）。
- **學習速度遷移：沒有。** first-batch 準確率全程持平（~0.15–0.20），晚段任務並沒有學得更快，沒有「learning to learn」。好處只體現在 asymptote（每個任務的 head 都是新的，前幾批還是得從頭訓）。

所以第三個目標**部分達成**：累積到更好的*表徵*（拉高新任務的天花板），但沒有更快的*習得過程*。嚴格分離「正向遷移」與「網路單純成熟」的對照是「task-N-在串流中 vs task-N-單獨訓練」——in-stream 的 0.94 遠高於單一任務新網路的 ~0.70，已是強證據，但該對照可把它釘死。

## 11. 誠實的硬測試：Class-IL（不給任務 ID）

§5–§10 的好結果幾乎都在 Task-IL（label_permuted，多頭、測試時用 task ID 選 head）。對話 §4 明講 **Class-IL（不給 task ID、單頭、在不斷增長的全域類別空間上分類）才是誠實的硬測試**。本節把這個拐杖拿掉。

### 11.1 一個良好定義的 Class-IL benchmark（sum-slice）

第一版嘗試「permute 輸入 + offset 標籤」是**退化的**：單一 permuted 輸入無法可靠辨識自己屬於哪個任務，全域標籤本質上模糊，連 Joint 上界都卡在 8%。改用 **sum-slice 設計**：把 sum 切成 `10×n_tasks` 個等頻細桶，task t 擁有一段連續的細桶（＝一段不重疊的 sum 範圍）。於是每個輸入的全域標籤是 sum 的確定性函數（無歧義、不需 task ID），而不同任務的輸入分布天然可區分（低 sum vs 高 sum）。

### 11.2 結果：Task-IL 的勝利不遷移（20 tasks＝200 classes，3 seeds）

| 方法 | final | forget | retention |
| :--- | :---: | :---: | :---: |
| Naive | 0.005 | 0.760 | 0.006 |
| **Joint（上界）** | **0.742** | — | — |
| Replay | 0.157 | 0.632 | 0.208 |
| **ReplayEWC** | **0.312** | 0.503 | 0.406 |
| DarkReplayEWC（DER++, α=0.5） | 0.232 | 0.584 | 0.300 |
| DarkReplayEWC（DER++, α=0.1） | 0.273 | 0.542 | 0.354 |

三個關鍵發現：

1. **Class-IL 殘酷得多。** 最佳法只有 0.312，離線上界卻有 0.742——差距 0.43，遺忘嚴重（retention 0.41）。對比 Task-IL（DER++ 幾乎貼到上界、retention 1.0），同一個 benchmark 換成不給 task ID 就幾乎是另一個問題。Naive 直接崩到 chance（recency bias：只有最新類別拿到正梯度）。
2. **DER++ 反轉成「有害」。** 在 Task-IL 是最佳加成（+7~14 分），在 Class-IL 卻**低於**純 ReplayEWC（α=0.5→0.232、α=0.1→0.273，都 < 0.312），而且 α 越小越好（越接近關掉蒸餾）。機制上：Class-IL 的單頭必須持續**重塑**輸出幾何以塞進新類別，而 logit 蒸餾把輸出錨向寫入當下的舊 logits（那時新類別還「不存在」），等於鎖死舊的類別邊界、加重 recency bias。Task-IL 每個 task 各有 head，蒸餾只穩定自己的 head、沒有跨類別競爭，所以才有益。
3. **EWC 才是跨 regime 穩健的核心。** ReplayEWC（0.312）遠勝純 Replay（0.157）——Fisher 選擇性權重保護在 Task-IL 與 Class-IL 都有效；DER++ 只在 Task-IL 有效。

### 11.3 解法：NCM 原型分類器把 Class-IL 缺口幾乎補滿

§11.2 指出問題在線性頭的 recency / magnitude bias。直接換掉讀出方式：**不用線性頭，改用「特徵空間最近類別原型（nearest-class-mean, cosine）」分類**，對所有看過的全域類別、不給 task id——這就是 iCaRL 的核心。表徵仍由 replay + EWC 學習；prototypes 從 buffer 算（類別平衡、無偏）。實作為 `NCMReplayEWC`（= HippocampalReplayEWC 的純 NCM 設定）。

| Class-IL, 200 classes, 3 seeds | final | forget | retention |
| :--- | :---: | :---: | :---: |
| ReplayEWC（線性頭） | 0.312 | 0.503 | 0.406 |
| Joint（線性頭，上界） | 0.742 | 0.116 | — |
| **NCMReplayEWC（原型讀出）** | **0.858 ± 0.003** | **0.060** | **1.07** |

- **缺口幾乎補滿**：0.312 → **0.858**，遺忘從 0.50 砍到 0.06，retention >1.0（BWT 甚至為正）。而且 3 seeds 方差極小（±0.003）。
- **甚至超過線性頭的 Joint 上界（0.742）**：因為線性 softmax 在 200 類上本身就受 recency/magnitude bias 拖累，而 NCM 在這個由 sum 結構化、近似一維有序的特徵流形上幾乎是最優讀出（公平的上界應是 Joint+NCM）。
- **這正好對映 §11.2 的機制診斷**：Class-IL 的病灶是「讀出層的偏置」，不是表徵本身——replay+EWC 學到的表徵其實夠好，換成無偏讀出就解放了。也解釋了為何 DER++（鎖死線性頭的輸出幾何）在這裡有害。
- **benchmark 的規模上限**：K=8 的位數和只有 ~73 個相異值，無法切成 >~200 個非空細桶（400 類時尾端桶被掏空）。所以 200 類（20 tasks）是本 benchmark Class-IL 的天然上限；要更大需調大 K。這是 benchmark 限制、不是方法限制。

**修正後的結論（回答「是不是找到持續學習演算法」）：在本 benchmark 上，Task-IL 與 Class-IL 都已有接近上界的配置**——Task-IL 用 Replay+EWC+DER++（retention~1.0），Class-IL 用 **Replay+EWC + NCM 原型讀出**（0.858、遺忘 0.06）。**但關鍵教訓是「對的機制隨設定而變」**：DER++ 在 Task-IL 是英雄、在 Class-IL 是負擔；真正跨設定穩健的是 replay + Fisher-EWC 當表徵學習骨幹，再依設定換上對的讀出（Task-IL 用 head + 蒸餾，Class-IL 用無偏原型）。距離「通用持續學習」仍有 §11.4 之後的硬牆（任務真正衝突、無 buffer、無邊界、規模）。

## 12. 結論

1.  **資料流設計**：pi 數位序列能為持續學習提供可重現、非重複的數據流，但「預測下一位」本質不可學，必須改用「窗口求和分桶 + 標籤隨機排列」。
2.  **標籤衝突之解決**：在 80 個任務的超長標籤重映射下，必須採用多頭結構（Task-IL）方能打破單輸出頭帶來的數學矛盾，使 HippocampalReplayEWC、SurpriseReplayEWC、ReplayEWC、Experience Replay 與 EWC 的全域平均準確率顯著攀升至 50% 以上，其中 HippocampalReplayEWC 已提升到 86% 以上。
3.  **機制互補性**：ContinualBP 強於可塑性維護，EWC 與 Replay 強於舊記憶維持。ReplayEWC 顯示樣本級 replay 與 shared-weight EWC 可以互補；ReplayContinualBP 則提醒我們，神經元重置若沒有更細緻的保護機制，仍可能傷害長期記憶。
4.  **腦啟發機制的邊界**：海馬迴式 episodic memory 對 Task-IL 有幫助，但必須 gated；固定比例記憶混合或過度 sleep consolidation 會在長流中傷害表現。
5.  **上界基準改寫了結論**：補上 Joint 離線上界後才看清，`label_permuted` 的真天花板是 99.3%（最佳法 86.6% 還有 ~13% 空間），而 `input_permuted` 連上界都只有 12%——後者根本不是遺忘問題，是單一共享輸入層的結構性矛盾。
6.  **轉接器把 input_permuted 從不可能變成可解**：給每個 task 一個輸入轉接器後，瓶頸從「結構不可學」轉成「一般的穩定性–可塑性問題」；接好 adapter 的 ReplayEWC 把 final average accuracy 從 12% 抬到 54.7%（上界 81.1%）。值得注意的是此 regime 下 **ReplayEWC 反而勝過 Surprise/Hippocampal**，與 `label_permuted` 的排名相反——說明前沿機制的優劣會隨任務結構翻轉，沒有單一全勝的方法。下一步是更大 buffer / 更強 shared-layer 保護來收掉剩下的 55%→81% 遺忘缺口，以及把 adapter 推向 learned router / modularity。
7.  **可永續學習的瓶頸是遺忘、不是可塑性流失（§9）**：把串流拉到 250 個任務，有 replay 的方法對角線不降反升，可塑性根本不綁定；專門維護可塑性的 `SustainableReplayEWC` 雖然完成設計目標（dead-unit≈0、effective rank 最高、且修好了 ReplayContinualBP 的記憶損傷），卻換不到準確率，因為這個 regime 本來就不缺可塑性。要再提升「可永續性」應主攻遺忘（更強記憶/正則、Benna-Fusi、generative replay）；可塑性注入要見效得換到無 replay 的 regime。
8.  **Benna-Fusi 複雜突觸：冪律遺忘有效，但棲位在 replay-free（§9.4）**：純 `BennaFusi` 把遺忘從 0.23 壓到 0.03，驗證冪律記憶有效；但它各向同性地拖住所有權重，在有 replay 時被 Fisher-selective 的 ReplayEWC 完全支配。它的真正價值是 replay-free / Fisher-free 的線上鞏固——80-task 下把 Naive 的 29% 翻倍到 53%、逼近 EWC 的 58% 且遺忘更低，且完全不存樣本、不算 Fisher。這把「該用哪種穩定機制」收斂成一條清楚的設計準則：能存樣本就用 replay(+EWC)，不能存樣本才換複雜突觸。
9.  **直攻遺忘的答案：函數空間錨定（DER++ logit 蒸餾）（§10）**：遺忘是函數空間干擾。混合式 `FunctionSpaceReplay` 的消融顯示，**GPM 梯度投影不適配平穩輸入的本 benchmark（凍結共享層）**，但 **DER++ logit 蒸餾**有效——在 130-task 串流上把 final 從 0.815 抬到 **0.892**、淨遺忘 retention→**1.01**、BWT→0、方差砍 4 倍，且優勢隨串流變長複利放大（250 tasks 領先 +14.2 分、retention 仍 0.97）。**Replay + Fisher-EWC + DER++ 蒸餾**是本專案目前最有效、且抗遺忘隨串流增強的演算法，也是設計教訓：對「函數」做錨定（輸出/logits）比對「權重」或對「輸入子空間」做約束更貼合本問題的遺忘來源。
10. **Class-IL 校正並隨後補強（§11）**。拿掉 task ID（單頭、200 類）後，Task-IL 的英雄 **DER++ 反轉成有害**（0.23 < 純 ReplayEWC 0.31），病灶是線性頭的 recency/magnitude bias。**把讀出換成 NCM 原型分類器（iCaRL 式，`NCMReplayEWC`）後，缺口幾乎補滿：0.31 → 0.858、遺忘 0.50 → 0.06、retention >1.0**（甚至超過線性頭 Joint 0.742）。關鍵教訓：**對的機制隨設定而變**——跨設定穩健的是 replay + Fisher-EWC 當表徵骨幹，再依設定換對的讀出（Task-IL：head + DER++ 蒸餾；Class-IL：無偏原型）。benchmark 的 Class-IL 規模上限 ~200 類（K=8 位數和僅 ~73 個相異值）。

---

## 檔案說明

- `pi_digits.py`：產生/快取 pi 小數位序列。
- `benchmark.py`：Permuted-Pi-Digits 串流（支持多頭 `label_permuted` 和單頭 `input_permuted`）。
- `model.py`：支持多頭選擇、per-task 輸入轉接器（`input_adapter`）與可塑性診斷的 numpy MLP 實現。
- `trainers.py`：16 種 CL Trainer，含 Naive、**Joint 離線上界**、EWC、（Task-Balanced）Replay、ReplayEWC、DarkReplayEWC、SurpriseReplayEWC、MarginSurpriseReplayEWC、HippocampalReplayEWC、ContinualBP、ReplayContinualBP、**SustainableReplayEWC（Fisher 保護的神經元回收）**、**BennaFusi / BennaFusiReplay（多時間尺度複雜突觸）**、**FunctionSpaceReplay（Replay + DER++ 蒸餾 + GPM 投影，可切換）**。所有 replay/記憶路徑都已接好輸入轉接器（依 task 分組套用對應 adapter）。
- `run.py` / `run_one_combo.py`：主實驗腳本（命令行選模式、方法、seed、任務數；`--input-adapter` 開啟輸入轉接器，`--joint-batch/--joint-steps` 控制上界）。
- `analyze.py`：彙整多 seed 實驗結果，輸出 JSON 與畫圖；缺 matplotlib 時用 Pillow 輸出圖表並產生 summary JSON。
- `results_label_permuted.json` / `results_input_permuted.json`：9 種主方法的原始數據。
- `results_joint_*.json`：Joint 離線上界。`results_naiveadapter_*.json` / `results_jointadapter_*.json` / `results_adapter_replay_*.json`：輸入轉接器系列實驗。
- `results_longstream_*.json`（130 tasks）/ `results_verylong_*.json`（250 tasks）/ `results_longstream_sustainable.json`：§9 可永續學習長串流探針數據。
- `results_bennafusi_dt0*_label_permuted.json`：§9.4 Benna-Fusi 複雜突觸 replay-free 定位實驗。
- `results_derpp_longstream_label_permuted.json`（130 tasks）/ `results_derpp_verylong_label_permuted.json`（250 tasks）：§10 函數空間 / DER++ 抗遺忘實驗。
- `results_classil_label.json` / `results_classil_dark01.json`：§11 Class-IL（誠實硬測試）實驗。
- `results_classil_ncm.json` / `results_classil_ncm_trainer.json`：§11.3 NCM 原型分類器解 Class-IL 缺口。
- `summary_stats_label_permuted.json` / `summary_stats_input_permuted.json`：跨 seeds 彙整後數據。
- `fig1_diagonal_accuracy_*.png` / `fig2_bwt_finalacc_*.png` / `fig3_plasticity_diagnostics_*.png`：主方法性能對比與診斷圖表。
- `fig4_input_permuted_adapter_ladder.png`：輸入轉接器打破結構性下限的階梯圖。
