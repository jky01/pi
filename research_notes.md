# 基於最新論文的持續學習演算法改進建議

這份文件針對目前 Permuted-Pi-Digits 持續學習框架中各模組的表現與限制，結合 2024 至 2026 年最新學術界研究成果（如 SERS、PCGR、自適應優化與參數高效能路由等），提出具體的改進建議。

---

## 1. P2.6 Regime & Horizon Detector (自適應任務流屬性檢測與蒸餾調節)

### 📌 背景與痛點
目前的實驗結果（`report.md` §12.3）顯示，**DER++ (Dark Experience Replay)** 的 logit 蒸餾在共享規則的長流（`label_permuted`）中能將保留率（retention）推上 1.01 的天花板；但在任務底層函數真衝突（`conflicting`）或短流時，固定的蒸餾會限制共享層更新，嚴重損害模型的可塑性。目前的局部安全閥（如梯度 cosine、信心門檻、延遲開啟等）屬於反應式（reactive）或過於保守，無法完美兼顧兩種任務流。

### 💡 最新論文啟發 (2024–2025)
*   **SERS (Self-Evolving Pseudo-Rehearsal, 2025)**: 提出使用動態正則化器（基於 Wasserstein 距離或任務特徵分布相似度）來動態調節不同任務間的約束強度。
*   **NCCL (Nonconvex Continual Learning, 2024)**: 透過前瞻性的梯度相似度評估來動態調整學習步長與正則懲罰。

### 🛠️ 具體改進建議（線上虛擬前瞻探針）
不要再微調單一 `alpha` 閥值，而是引入一個**反事實虛擬 SGD 前瞻（Counterfactual Lookahead Probing）**機制：
1.  **梯度前瞻計算**：在每個 step，利用目前 batch 的資料計算一個虛擬的權重更新 $\theta' = \theta - \eta \nabla L_{\text{current}}$。
2.  **計算衝突係數（Conflict Index）**：在 Replay Buffer 抽出的樣本上，計算更新前後的 loss 變化量：
    $$\Delta L_{\text{replay}} = L_{\text{replay}}(\theta') - L_{\text{replay}}(\theta)$$
3.  **自適應調整蒸餾強度 ($\alpha_{\text{adaptive}}$)**：
    *   若 $\Delta L_{\text{replay}} \le 0$（代表當前學習方向與歷史記憶**正向對齊**或無害）：將蒸餾係數拉滿（`alpha = 0.5`），實施 Proactive Consolidation，固化軟性幾何。
    *   若 $\Delta L_{\text{replay}} > 0$（代表當前學習與歷史記憶**嚴重衝突**）：動態降低蒸餾強度（`alpha -> 0`），釋放共享層表徵能力以適應新任務。
4.  **優勢**：此方法不需要維護雙軌 shadow model 的完整參數量，僅在每一步做一次反事實 gradient step，能精準捕捉「當前梯度是否會破壞舊記憶」的即時因果關係。

---

## 2. Class-IL 的原型漂移校正與高維流形匹配 (Prototype Optimization)

### 📌 背景與痛點
在 Class-IL（`class_il` 模式）下，線性輸出頭會遭受嚴重的 recency bias 破壞。我們雖然透過 `NCMReplayEWC`（原型分類器最近鄰讀出）把準確率從 31.2% 推上 85.8%（超過 Joint 線性頭），但在極長任務流中，隨著 Shared Backbone 持續更新以學習新類別，早期儲存在 Buffer 中的樣本其**特徵表徵會發生漂移（Representation Drift）**，導致舊的原型中心逐漸失效。

### 💡 最新論文啟發 (2025)
*   **PCGR (Prototype-Conditioned Generative Replay, NAACL 2025)**: 指出在持續學習中，模型參數的漂移會導致舊表徵與當前特徵提取器的空間錯位。該論文提出 **Prototype Shift Estimation (PSE)** 機制，透過估計特徵提取器的權重變化來修正歷史原型。

### 🛠️ 具體改進建議
1.  **原型漂移估計 (Prototype Shift Estimation)**：
    *   當我們在 task $t$ 訓練結束後，舊類別 $c$ 在 Buffer 中存有樣本 $\{x_i^c\}$。
    *   我們不需要重新對所有舊樣本跑 forward，而是利用 Buffer 裡少量的 Exemplars 重新計算當前特徵 $f_\theta(x_i^c)$。
    *   計算漂移向量 $\Delta \mu_c = \mu_c^{(\text{new})} - \mu_c^{(\text{old})}$。
    *   更新特徵空間中的原型中心：$\mu_c \leftarrow \mu_c + \Delta \mu_c$。
2.  **協方差感知原型匹配 (Covariance-Aware / Mahalanobis Prototype Matching)**：
    *   目前 NCM 採用餘弦相似度（Cosine Similarity）或歐氏距離，隱式假設特徵空間的分布是等向的（Isotropic）。
    *   建議在 Buffer 內為每個類別維護一個運行的**協方差矩陣對角線（Running Diagonal Covariance）** $\sigma_c^2$。
    *   在推論時，改用馬氏距離（Mahalanobis Distance）或高斯對數似然進行分類：
        $$\text{dist}(f(X), \mu_c) = \sum_{d} \frac{(f(X)_d - \mu_{c,d})^2}{\sigma_{c,d}^2}$$
    *   這能更細緻地刻畫 pi 數位求和分桶在隱藏層表徵空間中的長條狀或橢圓形幾何分布。

---

## 3. Domain-IL (input_permuted) 的自適應無邊界路由 (Task-Free Adapter Routing)

### 📌 背景與痛點
在 `input_permuted` 中，由於單一共享輸入層無法處理 80 種輸入隨機排列，我們實作了 `per-task input_adapter`（§7.2），成功解鎖了可學性（12% -> 55%）。然而，這套機制依賴測試時必須輸入 `task_idx`。如果在標準 Domain-IL 或 Task-Free 設置下，**測試時不提供 Task ID**，此機制便會失效。

### 💡 最新論文啟發 (2024–2025)
*   **Parameter-Efficient CL (PECL) & MoE Routing**: 近年針對 LoRA/Adapters 的持續學習研究，多數採用「Key-Query 匹配路由器」或「Entropy-based selection」。藉由將輸入特徵與一組特徵鍵值（Keys）比對，自動路由到最合適的子模組。

### 🛠️ 具體改進建議（Adapter 鍵值路由器）
將任務專屬的 Adapter 擴展為無 Task ID 的**動態門控轉接器（Gated Router Adapters）**：
1.  **維護 Task Keys**：為每個 task 的轉接器 $A_t$ 維護一個輸入空間的質心（Centroid / Key）向量 $K_t \in \mathbb{R}^{\text{in\_dim}}$，即該任務訓練資料認的平均 One-hot 激活分布（在 `input_permuted` 下，由於 Permutation 不同，特徵的邊際分布有著獨特的統計特徵）。
2.  **動態路由（Test-Time Routing）**：
    *   當測試樣本 $X$ 到來時，計算其與所有 Keys 的距離或餘弦相似度。
    *   選擇最接近的 $K_{\hat{t}}$，並將 $X$ 送入對應的 $A_{\hat{t}}$；或者使用 Softmax 權重進行混合：
        $$X' = \sum_{t} \text{softmax}(\text{sim}(X, K_t)) \cdot (X @ A_t)$$
3.  **線上學習 Key**：在訓練新任務時，以指數移動平均（EMA）在線上更新目前任務的 Key，完全不需要顯式的任務邊界宣告。

---

## 4. Task-Free CL 與線上 Fisher 估計 (Online Continuous EWC)

### 📌 背景與痛點
目前 EWC 正則化高度依賴 `on_task_end` 勾子函數。當任務結束時，模型需要停下來，掃描整個任務的訓練集以計算 Fisher 資訊矩陣並鎖定 Anchor。這在真實的「無邊界持續學習（Task-Free CL）」中是不可能的。

### 💡 最新論文啟發 (2024)
*   **Online Regularization**: 提出在訓練過程中即時更新參數敏感度，而非在任務邊界進行批次計算，從而將算法無縫推廣至 Task-Free 串流。

### 🛠️ 具體改進建議
1.  **線上 Fisher 估計 (Online EMA Fisher)**：
    *   在每個 step 的 backward 過程中，在取得梯度 $g = \nabla_\theta L$ 後，直接以指數移動平均更新 Fisher 對角線：
        $$F \leftarrow (1 - \rho) F + \rho g^2$$
        其中 $\rho$ 是極小的衰減率（例如 $10^{-4}$）。
2.  **滾動 Anchor 更新 (Rolling Anchor / Weight Decoupling)**：
    *   不再於任務結束時突變式地更新 Anchor，而是將參數錨定限制轉換為一種**軟約束阻尼**，讓 Anchor 以極慢的速度往當前參數 $\theta$ 漂移，或者在線上偵測到「Surprise（損失函數突增）」時才觸發 Anchor 局部更新。
3.  **Surprise-based Boundary Detector**：
    *   監控訓練損失的指數移動平均 $\bar{L}$。當發現短期損失顯著高於長期均值（即 $\text{Loss} > \text{Threshold} \times \bar{L}$）時，自動判定為分布漂移，觸發一次內部的 EWC 鞏固更新。

---

## 5. 記憶空間優化：特徵級重播與偽回放 (Feature Replay & Pseudo-Rehearsal)

### 📌 背景與痛點
雖然 ReplayEWC 表現優異，但儲存 2000 個原始樣本（每個 $80$ 維 one-hot 向量）在極長串流中仍會佔用實體記憶體，且在實際應用中（如隱私敏感數據）儲存原始輸入可能不被允許。

### 💡 最新論文啟發 (2024–2025)
*   **Prototype-Conditioned VAE / Generative Replay (2025)**: 顯示利用統計原型生成的偽特徵（Pseudo-features）進行重播，其效果已能媲美甚至超越儲存原始樣本。

### 🛠️ 具體改進建議
1.  **特徵重播 (Feature Replay / Mid-Level Replay)**：
    *   不儲存 80 維的原始輸入 $X$。我們只儲存第二層隱藏層的激活特徵 $a_2 \in \mathbb{R}^{64}$ 與其對應的 Label。
    *   當需要重播時，直接將 $a_2$ 注入到第三層（線性頭），繞過前兩層 MLP。這大幅減少了梯度傳播的計算量，且儲存維度更小。
2.  **高斯偽表徵回放 (Gaussian Pseudo-Rehearsal)**：
    *   徹底捨棄 Replay Buffer。
    *   在特徵空間中，為每個類別 $c$ 紀錄其均值 $\mu_c \in \mathbb{R}^{64}$ 與方差對角線 $\sigma_c^2 \in \mathbb{R}^{64}$。
    *   訓練新任務時，從歷史類別的高斯分布 $\mathcal{N}(\mu_c, \text{diag}(\sigma_c^2))$ 中隨機採樣偽特徵，並與當前任務的真實特徵一起輸入給輸出端（線性頭或原型分類器）進行聯合更新。
    *   這能實現真正的 **Buffer-free**，並將記憶體限制縮小為常數級 $\mathcal{O}(\text{n\_classes} \times \text{feat\_dim})$。

---

## 📊 改進方案優先級與預期效益 scorecard

| 建議方案 | 針對痛點 | 預期效益 | 實作難度 | 推薦順序 |
| :--- | :--- | :--- | :---: | :---: |
| **1. 反事實前瞻探針 (P2.6)** | DER++ 在 conflicting 拖累 final/Joint | 兼顧長流抗遺忘與短流/衝突任務的可塑性 | 中等 | **1 (P2.6 首選)** |
| **2. 原型漂移校正 (PSE)** | Class-IL 長流特徵漂移 | 穩定 200 類以上極長 Class-IL 串流的表現 | 簡單 | **2** |
| **3. 鍵值路由轉接器** | Domain-IL 測試時需要 Task ID | 實現無 Task ID 的 Permuted Domain 學習 | 中等 | **3** |
| **4. 線上 Fisher 估計** | EWC 依賴 task_end 邊界 | 走向真正的 Task-Free 無邊界持續學習 | 中等 | **4 (P3 必備)** |
| **5. 高斯偽表徵回放** | Buffer 儲存空間與隱私限制 | 達成 Buffer-free，降低儲存空間與計算開銷 | 簡單 | **5 (P4 必備)** |
