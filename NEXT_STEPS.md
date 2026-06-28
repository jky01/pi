# NEXT_STEPS.md — 未完成工作與後續計畫（給下次運行參考）

這份檔案記錄「還沒做的工作項目 + 為什麼做 + 預計怎麼做」，讓下次運行（cold start）能直接接續，不必重新推導已有結論。完成一項就把它從 backlog 移到「已完成」並更新 `report.md`。

---

## 0. 環境與如何跑（重要，先讀）

- **Python 直譯器**：系統預設的 `python` / `python3` **沒有 numpy**。要用這支（有 numpy 2.3.5，且 P8 起已裝 torch 2.12.1 / torchvision 0.27.1 CPU）：
  `/Users/jackyyeh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`
- **P8 特徵管線（torch）**：`$PY extract_features.py` 一次性抽好 CIFAR-100 的 frozen ResNet18 特徵（存 `cifar100_resnet18.npz`，已 gitignore），之後 `$PY run_features.py ...` 純 numpy 在特徵上跑 CL，重用全部 trainers。
- 跑 benchmark：`$PY run.py <mode> --methods ... --seeds 0 1 2 --n-tasks N --steps-per-task S --output results_xxx.json`
  - mode ∈ `label_permuted`（Task-IL 多頭）、`input_permuted`（Domain-IL 單頭，加 `--input-adapter`）、`class_il`（單頭、無 task id）。
- 冒煙測試（目前 registry 內所有 trainers × label/input/conflicting 三種 mode）：`$PY -m unittest test_smoke`
- 長串流受 `pi_digits_600000.txt` 600k 位數限制：`n_tasks*(steps+test) ≲ 560k`。
- 結果分析：`$PY analyze.py <mode>`，或直接讀 `results_*.json`（`run.summarize` 算 final/bwt/forget/retention）。

### 0.1 在另一台機器接續（含 NVIDIA CUDA GPU，如 RTX 2070）
- **遷移方式＝git**：`git clone https://github.com/jky01/pi.git` 後 `git checkout p8-pretrained-cl`（最新進度在這條 branch）。`pi_digits_600000.txt` 已在 repo 內；大檔（`mnist.npz`/`cifar100_resnet18.npz`/`cifar_data/`）是 gitignore、**會自動重新下載/重生**，不必搬。
- **環境**：新機自建 venv，`pip install numpy torch torchvision`（裝 **CUDA build** 的 torch，非 CPU build）。把上面寫死的 mac 直譯器路徑換成新機的 `python`。
  - **本 RTX 2070 機器已建好**：`PY=/home/aa/pi/.venv/bin/python`（Python 3.14、numpy 2.5.0、torch 2.12.1+cu130 / torchvision 0.27.1+cu130，`torch.cuda.is_available()=True`，device=NVIDIA GeForce RTX 2070, sm_75, 8GB）。系統 `python3` 只有 numpy、無 torch。CIFAR-100 raw 已下載在 `./cifar_data/`。
- **資料重生**：`python mnist_data.py`（自動下載 MNIST）、`python extract_features.py`（自動下載 CIFAR-100 + ResNet18 權重並抽特徵）。
- **用 GPU 跑（唯一 torch 訓練實驗）**：`python run_backbone_transfer.py --device cuda ...`（或 `--device auto`，會自動選 cuda）。`pick_device` 已支援 `cuda > mps > cpu`；其餘 numpy 實驗不吃 GPU。
- **接手脈絡**：先讀本檔 + `report.md` §0（TL;DR）+ §17–§23（P7–P9 現代化弧線）。`report.md` 是完整研究日誌，整段對話的結論都已落在裡面——新開的 cold-start session 讀這幾份即可無縫接續。

## 1. 已確立的結論（不要重做，直接引用 report.md）

- **瓶頸是遺忘、不是可塑性流失**：有 replay 時對角線到 250 tasks 仍上升（§9）。
- **Task-IL 最佳配置**：`Replay + Fisher-EWC + DER++ logit 蒸餾`（= `DarkReplayEWC --dark-alpha 0.5`），130/250-task retention ≈ 1.0、BWT≈0（§10）。
- **input_permuted 需要 per-task 輸入 adapter**（`--input-adapter`），12%→55%（上界 81%，§7）。
- **Benna-Fusi / SustainableReplayEWC**：機制有效但只在 **replay-free** regime 才有價值；有 replay 時被 Fisher-EWC 支配（§9.4）。
- **GPM 梯度投影不適配本 benchmark**（輸入平穩→凍結共享層）；函數空間錨定要用「輸出蒸餾」不是「輸入子空間投影」（§10.1）。
- **Class-IL（誠實硬測試，§11）**：拿掉 task id 後線性頭因 recency bias 崩壞（**DER++ 反轉成有害**）。**已解：`NCMReplayEWC`（最近類別原型讀出，iCaRL 式）把 0.31→0.858、遺忘 0.50→0.06、retention>1.0**（§11.3）。教訓：表徵骨幹用 replay+EWC，讀出依設定換（Task-IL：head+DER++；Class-IL：無偏原型）。benchmark Class-IL 上限 ~200 類（K=8 位數和僅 ~73 相異值）。
- **Adaptive distillation 初步結果（§12.2）**：`AdaptiveDarkReplayEWC` 可在 conflicting 下自動把 α 降到 0，退回 ReplayEWC、避免固定 DER++ 傷害；confidence gate 可減少低品質 logits 的副作用。但目前梯度 cosine 只能當安全閥，還不能自動判斷何時該在長流共享規則下打開 DER++。
- **Pressure / maturity gating 邊界（§12.3）**：`PressureDarkReplayEWC`（可靠 logits × label-loss pressure）安全但偏保守，130-task 把 ReplayEWC **0.815→0.847**，仍低於 full DER++ **0.892**；logit drift 是假警報，delayed DER++ start=40/80 也不夠。下一步要做的是 **regime/horizon detector**，不是再調單一 batch-level gate。
- **cos-RTP regime detector 失敗（§12.4，3-seed 驗收）**：task-onset 共享層梯度餘弦**不是**有效 regime 訊號——RTP 幾乎永遠判 conflicting、退化成 ReplayEWC，130-task 只有 **0.818**（< full DER++ 0.893）。機制完好（強制永遠開→0.898），病灶在訊號（多頭標籤排列污染共享層梯度方向）。教訓：局部/reactive/權重空間訊號（cosine、logit drift、label loss）都不足，要用 function-space 反事實量測或 horizon 訊號（→ P2.7）。
- **P2.7 horizon oracle 驗證（§12.5，部分正面）**：`HorizonDarkReplayEWC` 證明「已知 horizon」可作上界 schedule：20-task label 關閉 DER++ 得 **0.873**（= ReplayEWC）、40-task conflicting 關閉得 **0.486**（= ReplayEWC）、80/130-task label 開啟得 **0.868/0.893**（= full DER++）。但 80-task × 2000 steps 反例顯示 horizon 長度本身不夠，還要看訓練 budget / function-space benefit。
- **P2.8 online function-space benefit detector（§12.6，部分正面/負面）**：`BenefitDarkReplayEWC` 用可回復虛擬步比較 DER++ on/off。label-loss benefit 版成功避開誤開：conflicting 40 **0.486**、80×2000 **0.788**，都等於 ReplayEWC；但長流 80/130 也只等於 ReplayEWC（**0.818/0.832**），拿不到 full DER++ **0.868/0.893**。logit-MSE benefit ablation 會開，但在短流/80×2000 誤開並貼近 DarkReplayEWC 壞結果。教訓：**單步反事實太短視；logit 幾何太假陽性；DER++ 的收益是多步、慢時間尺度的 proactive consolidation。**
- **P2.9 local multi-step slow-benefit controller（§12.7，負面結果）**：`SlowBenefitDarkReplayEWC` 把 P2.8 probe 擴成最近 5 個 current batches 的 shadow rollout。label-loss 版保持安全：20-task **0.873**、80×2000 **0.788**、conflicting 40 **0.486**，但 alpha 仍全程 0，80/130 長流仍是 **0.818/0.832**，拿不到 full DER++ **0.868/0.893**。小 logit 權重 0.02 會讓 80×2000 seed0 掉到 **0.771**。教訓：同一局部 window 多走幾步仍不是長期因果收益；下一步要做跨真實時間持續存在的 shadow/bandit 或顯式 regime prior。
- **Task-free 的代價趨近於零（§13，P3）**：把 Fisher/anchor 鞏固從 `on_task_end` 邊界觸發改成固定步距線上估計後，80-task 下 ReplayEWC 0.818→0.826、DER++ 0.868→0.855（在 std 內）。核心配方（replay + Fisher-EWC + DER++）天生接近 task-free；邊界在此 regime 可有可無。
- **Buffer-free 生成式回放長流反超 raw replay（§14，P4）**：`GenerativeReplayEWC` 不存原始樣本，label_permuted 80-task **0.916** 勝過 raw ReplayEWC 0.818 / DER++ 0.868（固定 buffer 在長流被稀釋、生成統計量不衰減）；但須條件在定義標籤的統計量上（sum-matched ablation 0.470→0.753），conflicting 多變規則下保真度不足（0.431<0.486）。
- **rule-agnostic 生成回放：scholar teacher 補上 conflicting（§15，P5）**：NB 自分類器失敗（太弱，label 退步到 0.521）；`ScholarGenerativeReplayEWC`（每任務凍結 teacher 對合成輸入蒸餾）conflicting 0.474（final/Joint 差 raw 僅 1.6%）、遺忘最低 0.082。沒有單一 buffer-free 生成器全勝（簡單統計量→sum-match；複雜規則→scholar）；殘留缺口源自因子化輸入保真度。
- **on-manifold 輸入生成器假設被推翻（§16，P6，負面）**：全域 per-task 邊際抽 on-manifold 輸入 + scholar，3-seed 全面更差（conflicting 0.428、label 0.769），瓶頸是可塑性。per-class 集中回放 > 全域覆蓋；殘留小差距是真實樣本的不可取代價值（正向後向遷移），非輸入失配。整個 P4–P6：不存原始樣本可行且常足夠（長流甚至贏 raw），但完全追平 raw 仍有一道由真實樣本聯合結構撐起的小硬牆。
- **正向遷移**：表徵層有（晚段任務最終準確率更高），學習速度沒有（§10.3）。

## 2. Backlog（依優先序；每項含 為什麼 / 做法 / 驗收）

### 工作狀態總覽（先看這裡）

**已完成 / 做完的事情**
- **P1 — Class-IL 遺忘缺口**：已用 `NCMReplayEWC` 解掉線性頭 recency bias，詳見 §11.3。
- **P2 — Conflicting-task benchmark**：已新增真衝突任務、40-task scorecard、DER++ alpha sweep，詳見 §12.1。
- **P2.5 — Pressure / maturity gating**：已實作 `PressureDarkReplayEWC`、confidence/loss-pressure gate、delayed DER++ maturity gate，並完成 20/80/130-task 對照，詳見 §12.3。
- **P2.6 — Lookahead 與 RTP 動態 Regime 偵測器（已做完並 3-seed 驗收，結論為負面）**：實作了 Lookahead 與 cos-RTP 門控，但 **3-seed 正式驗收推翻初版「成功」結論**：cos-RTP 在 130-task label_permuted 只有 0.818（< full DER++ 0.893、甚至略低於 ReplayEWC 0.832），因為它幾乎永遠判 conflicting、退化成 ReplayEWC（alpha_mean≈0.011）。threshold 掃描證明機制完好（強制永遠開→0.898）、病灶是訊號（task-onset 梯度餘弦分不開共享規則長流 vs 真衝突）。詳見 §12.4.1/12.4.2。**P2.6 的目標（自動辨識何時值得 proactive consolidation）尚未達成 → 退回 backlog 為 P2.7。**
- **P2.7 — Horizon oracle regime detector（已完成，§12.5，部分正面）**：實作 `HorizonDarkReplayEWC` 與 `--horizon-threshold/--horizon-override`。標準驗收下，oracle horizon gate 可在 20/40-task 關閉 DER++、在 80/130-task 開啟 DER++，同時貼近 ReplayEWC 安全性與 full DER++ 長流增益。限制：80-task × 2000 steps 顯示「任務數夠長」不一定代表 DER++ 立刻有益，下一步需改成 online function-space benefit probe。
- **P2.8 — Online function-space benefit detector（已完成，§12.6，部分正面/負面）**：實作 `BenefitDarkReplayEWC`、`--benefit-*` CLI、diagnostics。label-loss 版是安全閥：20-task label **0.873**、80×2000 **0.788**、conflicting 40 **0.486**，都避開 DER++ 傷害；但 80×4000 **0.818**、130×4000 **0.832**，仍退回 ReplayEWC，未達 ≥0.87 長流目標。logit-MSE benefit ablation 證明「保留舊 logits 幾何」不是可靠收益訊號，會在短流/未成熟長流誤開。
- **P2.9 — Local multi-step / slow-benefit controller（已完成，§12.7，負面結果）**：實作 `SlowBenefitDarkReplayEWC`、`--slow-rollout-steps`、diagnostics。recent-window 5-step label-loss probe 保持 P2.8 安全性但仍全程 alpha=0；小 logit 權重 0.02 仍誤開 80×2000。結論：local rollout 不是 DER++ 長期收益 detector，下一步若繼續此線，需做 persistent shadow-model bandit。

- **P3 — Task-free (無邊界) CL（已完成，§13）**：新增 `OnlineEWCReplay` / `OnlineDarkReplayEWC`，把 Fisher/anchor 鞏固從 `on_task_end` 邊界觸發改成固定步距線上估計（從 reservoir buffer 取樣）。80-task × 3 seeds：失去邊界知識的代價趨近於零（EWC +0.008、DER++ −0.012 在 std 內），無邊界 DER++ 仍勝過有邊界 ReplayEWC。
- **P4 — Buffer-free / generative replay（已完成，§14）**：新增 `GenerativeReplayEWC`（完全不存原始樣本，per-(task,class) categorical 生成模型 + sum-matched conditional generation）。label_permuted 80-task buffer-free **0.916 反超** raw ReplayEWC 0.818 與 DER++ 0.868（長流 buffer 被稀釋、生成統計量不衰減）；conflicting 0.431 < raw 0.486（sum-matched 條件統計量對非 sum 規則不符）。
- **P5 — rule-agnostic 生成回放（已完成，§15）**：試兩條路。`NBGenerativeReplayEWC`（從儲存 categorical 自建 NB 分類器 rejection）**失敗**（conflicting 0.418、label 退步到 0.521）。`ScholarGenerativeReplayEWC`（每任務凍結 teacher、對合成輸入 soft-logit 蒸餾，generative DER++）**成功補上 conflicting 缺口**：0.474（final/Joint 0.628 vs raw 0.644，差 1.6%≤2%）、遺忘最低 0.082；label_permuted ≈ raw（0.809）但不及 sum-match 峰值 0.916。沒有單一 buffer-free 生成器全勝；殘留缺口源自因子化輸入保真度 → P6。
- **P6 — 更強的輸入生成器（已完成，§16，負面結果）**：`ScholarGlobalGenerativeReplayEWC`（全域 per-task 邊際抽 on-manifold 輸入 + scholar 標註）**推翻 P5 的 on-manifold 歸因**——3-seed 全面更差（conflicting 0.428 < scholar-class 0.474；label 0.769 < 0.809），儘管 forgetting 最低（0.052），瓶頸是可塑性（diag 0.54→0.45）。per-class 集中回放比全域覆蓋更重要；殘留小差距更像真實樣本不可取代的價值（精確 per-class 聯合結構→正向後向遷移），非輸入分布失配。
- **P7 — 標準 benchmark 外部效度驗證（已完成，§17）**：純 numpy 載入真實 MNIST（`mnist_data.py`/`mnist_benchmark.py`/`run_mnist.py`，介面與 pi stream 一致、重用全部 trainers）。**Permuted-MNIST（多頭 Task-IL，20×3 seeds）：DER++ 函數錨定守住**——DarkReplayEWC **0.912** > ReplayEWC 0.896 > Naive 0.840、BWT→-0.002、retention 0.998、幅度隨串流複利。**Split-MNIST class-IL（5 tasks、10 類、無 task id）：NCM 降遺忘方向守住**（forget 最低 0.028、retention 最高 0.977）**但 pi 的「線性頭崩潰＋DER++ 反轉＋NCM 唯一解」被推翻為 200 類特例**（線性頭 0.926 不崩、DER++ 0.952 反而最佳）。教訓：核心機制跨 benchmark 成立，但戲劇性數字與「某機制必反轉」的強論斷隨類別數/串流長度/buffer 比例而變、不可外推。結果檔 `results_mnist_{permuted,split}.json`。
- **P8 — frozen pretrained 特徵上的 Class-IL（已完成，§18）**：裝 torch，用 frozen ImageNet ResNet18 抽 CIFAR-100 特徵（`extract_features.py`→`cifar100_resnet18.npz`），在特徵上維持純 numpy 訓練小 head（`feature_benchmark.py`/`run_features.py`，重用全部 trainers）。標準 **Split-CIFAR-100**（20 tasks×5 類、無 task id、3 seeds）：Naive **0.063**、線性頭 ReplayEWC **0.471（forget 0.440）**、DER++ **0.495**、**NCMReplayEWC 0.530（forget 0.198、retention 0.743）**。**三 benchmark 合看：class-IL「線性頭→NCM」改善隨類別數單調放大（10 類微弱→100 類大→200 類戲劇性）→ pi 的 NCM 英雄結論不是特例、是類別數的函數。** 強表徵把規模/表徵牆推近一步但牆沒倒（53% final 且是最有利設定）。結果檔 `results_feature_split_cifar100.json`。

**已停損（不再投入，理由見下）**
- **P2.10 — persistent shadow-model bandit / regime prior（停損）**：P2.5→P2.9 共 5 輪自動 DER++ gating 全部報酬遞減/負面，且都是在 pi 數位的特性上精雕細琢。P7 外部驗證（§17）顯示「該不該開 DER++」的答案本就隨類別數/串流長度/buffer 比例而變，這條線外部效度低。實務穩健解已知：長共享流直接開 full DER++、短流/衝突用 ReplayEWC（P2.7 horizon oracle 已證明可切換 schedule 存在）。**停止再做自動 detector。**
- **P6b — per-class autoregressive 生成器（停損）**：邊際空間僅 1.6% final/Joint，且 P6 已證明殘留缺口是「真實樣本不可取代的價值」而非輸入保真度，繼續投入價值低。**停損。**

- **P10 — 量測正向遷移 / 累積（已完成，§19）**：在 Split-CIFAR-100 frozen 特徵流上量「學新 task 的速度」（`run_forward_transfer.py`）：持續模型(ReplayEWC) vs 同特徵、隨機初始化、只學該 task 的 fresh head，用 task-k 受限 5-way acc 隔離新任務本身學多快。**結果：正向遷移≈0**——每個步數預算(2/5/10/20)持續模型都不比 fresh 快、反而略慢（Δ -0.037/-0.022/-0.008/-0.013），且不隨經驗增長。Naive-continual 對照 Δ≈0 分離出因果：**主因 frozen backbone 已封頂(沒東西可累積)、抗遺忘機制再加小幅可塑性稅**。量化了「最遠那道牆」：系統只會持續不忘、不會持續變強。結果檔 `results_forward_transfer_{cifar100,naive_cifar100}.json`。

- **P8b — backbone 也持續適應（已完成，§20，正面結果）**：用從零小 CNN、backbone 跨 task 持續適應（`run_backbone_transfer.py`，torch + MPS GPU），量「學新 task 的速度」。**三 regime 對照**：frozen(P10)→Δ≈0；會動但 **Naive→Δ 負(-0.044，重現 loss of plasticity)**；會動且 **Replay→Δ 強正(+0.174)且隨經驗單調增長**（early +0.035→late +0.282，per-task late +0.30~+0.40，2 seeds）。**「持續變強」終於出現,充要條件=表徵持續建構 × 可塑性持續維持。** 修正 P10 的悲觀結論。MPS micro-benchmark：小 CNN 訓練 CPU→MPS 約 5–6×（width64 batch32: 34.6→175 steps/s）。結果檔 `results_backbone_transfer_{cifar100,replay_cifar100}.json`。

- **P8c — 放大到真 ResNet18（已完成，§21，正面結果）**：把 P8b backbone 換成 CIFAR-adapted ResNet18（~11M、end-to-end、MPS）。正向遷移**更大且累積更陡**：Δ@40 overall +0.232（小 CNN +0.174）、late +0.356（小 CNN +0.282）、continual late 絕對 acc 0.73–0.75。**「持續變強」隨容量放大**，非小模型玩具效應。`run_backbone_transfer.py` 加 `--arch {smallcnn,resnet18}`。結果檔 `results_backbone_transfer_resnet18_cifar100.json`。

- **P8d — DER++ 維持機制 + 穩定-可塑性甜蜜點（已完成，§22，正面結果）**：在會動 ResNet18 上把 continual 維持機制從 plain Replay 換成 **DER++（replay CE + logit 蒸餾 α=0.5）**，同框架同時量正向遷移與 retention。**DER++ 兩軸全勝、無 tradeoff**：正向遷移 Δ@40 +0.268（replay +0.180）、retention mean_final 0.641（0.564）；兩者 mean_forgetting 皆負＝backward transfer。logit 蒸餾無可塑性稅、反而讓新任務學更快。§10 的最佳抗遺忘法升級成最佳累積學習法。`run_backbone_transfer.py` 加 `--continual-mode derpp --dark-alpha`。結果檔 `results_p8d_{replay,derpp}_resnet18.json`。

- **P9 — 無 buffer 下的累積（已完成，§23，負面結果）**：用 **LwF**（=DER++ 蒸餾項但不存樣本）做乾淨 ablation。buffer-free **完全失敗**：naive Δ@40≈0、**LwF 比 naive 更差**（-0.089，λ=0.1/1.0 皆然），continual 卡在亂猜、負向遷移隨經驗惡化（蒸餾錨點累積→凍結 backbone）。**乾淨結論：DER++ 的有效引擎是 replay-CE on 真實樣本，不是 logit 蒸餾；蒸餾只是放大器、單獨用反而凍結。** L3（會累積）達成但架在 replay buffer 拐杖上。`run_backbone_transfer.py` 加 `--continual-mode lwf --lwf-lambda`。結果檔 `results_p9_{naive,lwf,lwf_lam0.1}_resnet18.json`。

- **P9b — 無 buffer 累積的其他機制（已完成，§24，正面結果）**：P9 只證明「DER 式 logit-MSE 蒸餾(LwF)」不行；P9b 試另兩條 buffer-free 路，**兩條都成功**。實作在 `run_backbone_transfer.py`：(a) `--continual-mode pnn`（PNN 式參數隔離，smallcnn columns + lateral）→ Δ@40 **+0.101**、**mean_forgetting 0.000**、mean_final 0.501（介於 smallcnn naive −0.044 與 replay +0.174 之間；代價＝容量隨 task 線性成長 + 推論需 task-id 路由）；(b) `--continual-mode lwf_kd --lwf-temp 2.0`（經典 softmax-KD，resnet18）→ Δ@40 **+0.089**、early +0.010→late +0.159（GROWS）、mean_final **0.357**（≫亂猜 0.20、>naive 0.214，**不凍結**）——**推翻 P9 caveat**：「蒸餾必凍結」只是 DER 式 logit-MSE 的特例，softmax-KD 保住可塑性。兩條都仍低於 buffered replay/DER++（真實樣本仍獨佔「高絕對 retention + 後向遷移」）。結果檔 `results_p9b_{pnn_smallcnn,lwfkd_resnet18}.json`。(c) 生成式回放搬到會動 backbone 尚未做（最重、優先序低）。

- **P11 — 無 task-id 部署（class-IL）下的累積（已完成，§25，分支 `taskfree-accumulation`）**：把 P8b–P9b 的 task-IL 探針換成 class-IL（無 task id，100 類 argmax）。`run_backbone_transfer.py` 加 `--eval-mode classil`（+ NCM 原型讀出 `build_ncm`/`ncm_acc`，用 head/fc 的 forward hook 抽特徵）。resnet18、20 tasks、2 seeds：**DER++ task-IL 0.641 → class-IL 線性 0.177 / NCM 0.245**、Replay 0.564→0.107/0.170、Naive 0.214→0.031/0.062。發現:(1) task-id 撐住大部分數字;(2) 排序保留(DER++>Replay>Naive),骨幹必要但不充分;(3) NCM 在會動 backbone 上首次驗證可部分救回(~1.4–2× 線性、但只補一半);(4) 表徵累積本身活著(class-IL all-seen 探針 Δ replay/derpp 仍正且增長)→**瓶頸是 task-free 讀出/校準,不是表徵不累積**。設計注意:continual-vs-fresh 速度 Δ 在 class-IL 有 confound(fresh 無真競爭者),公平累積-速度比較仍用 task-IL 5-way。結果檔 `results_taskfree_{naive,replay,derpp}_resnet18.json`。

- **P12 —（承 P11）會動表徵上比 NCM 更好的 task-free 讀出（已完成，§26，部分負面/修正歸因）**：在同一批訓練好的會動 backbone 上加兩個 post-hoc 讀出——`cosine_acc`（L2-normalize 頭權重+特徵）與 `bic_fit`/`bic_acc`（per-task-group affine,class-balanced 校準集擬合）。resnet18、100 類、2 seeds:DER++ 線性 0.172 / NCM 0.235 / cosine 0.154 / BiC 0.216;Replay 0.132/0.210/0.091/0.179。**BiC 修一部分偏置(+~25% 相對)但封頂在 NCM 之下;cosine 有害;沒有讀出接近 task-IL 0.641。** 修正 P11 歸因:NCM 已是最乾淨讀出(最終特徵空間重算新鮮原型),0.235 代表**特徵本身分不開 100 類＝全域跨任務可分性不足**,讀出補不了。結果檔 `results_p12_{naive,replay,derpp}_resnet18.json`。

- **P13 —（承 P12）cross-buffer SupCon 直攻全域可分表徵（已完成，§27，負面結果）**：在會動 backbone 加 SupCon（`--supcon-weight`/`--supcon-temp`，跨 [當前∪replay] penultimate 特徵）。DER++、resnet18、class-IL、掃 supcon ∈ {0,0.5,1.0}:**三軸單調惡化**（NCM 0.237→0.209→0.194、forgetting 0.455→0.480、累積探針 Δ +0.234→+0.204）。病灶:replay batch 每類 <0.5 樣本、舊類湊不出同類正對 → SupCon 只收緊當前任務簇、又與 CE/DER++ 搶容量。**否證的是「在稀疏 replay batch 上直接做 SupCon」,不是 SupCon 概念**;與 §16/P4/P9 同一道「真實樣本/每類密度」硬牆。結果檔 `results_p13_derpp_supcon{0,0.5,1.0}_resnet18.json`。

- **P13b —（承 P13）prototype memory bank（proto-contrastive）（已完成，§28，診斷正確但仍負面）**：實作 `ProtoBank`（持久 per-class 原型,current+replay EMA 更新)+ `--proto-weight`/`--proto-temp`/`--proto-momentum`。DER++、class-IL 掃 proto ∈ {0,0.1,0.5,1.0}:NCM 0.237→0.233→0.221→0.207。對照 P13 SupCon 同劑量 0.237→0.209→0.194——**原型庫把傷害變小(證明 P13「正對稀疏」診斷正確),但仍無任何劑量淨增益**。結論:修好正對稀疏是必要不充分,auxiliary 對比目標抬不動 class-IL,瓶頸是表徵起點/容量。結果檔 `results_p13b_derpp_proto{0,0.1,0.5,1.0}_resnet18.json`。

- **P13c —（已完成，§29，正面結果）廣泛預訓 init + 會動 fine-tune**：`--pretrained`（layer1-4 載 IMAGENET1K_V1、stem/fc 新）。**控 lr confound**（預訓 lr0.05 被 fine-tune 打爛→NCM 0.090；故預訓與從零都用 lr0.01）。derpp、20t、100 類、2 seeds：從零 lr0.01 NCM **0.222**（≈lr0.05，降 lr 對從零無幫助）、**預訓 lr0.01 NCM 0.440**（線性 0.140→0.362、BiC 0.202→0.440）——**同 lr 下預訓近乎翻倍,是整條弧線最大單一槓桿**。表徵級 stability–plasticity 階梯：從零會動 0.237 → 預訓會動 0.440 → 預訓凍結(§18)0.530（fine-tune 侵蝕預訓的全域可分性）。結果檔 `results_p13c_{fromscratch,pretrained}_lr01_resnet18.json`。

**下一個要做 / Todo**
- **P14 —（承 P13c，最高優先）保留 plasticity 同時不侵蝕預訓全域可分性（表徵級 stability–plasticity）**：P13c 揭示真正的開放問題＝fine-tune(會動→task-IL 累積)會把預訓的 class-IL 可分性從 0.530 侵蝕到 0.440。候選:(a) **分層 lr / 凍結淺層**（低層保預訓通用特徵、只 fine-tune 高層）;(b) **預訓特徵蒸餾錨**（用凍結預訓 backbone 當 teacher 對特徵做距離正則,類似 LwF 但錨在預訓表徵而非舊任務 logits）;(c) **參數高效 fine-tune（LoRA/adapter）**只動少量參數、保住預訓 body。目標:把預訓 fine-tune 的 class-IL 從 0.440 往 frozen 0.530 甚至更高推,同時保住 task-IL 累積。這條直接對應 LLM 終身微調的核心張力。
- **（可選）P9c — 把 PNN 放大到 resnet18 + 結合 softmax-KD**：如 P8b→P8c 量容量效應。機制已確立,優先序中等。
- **（可選）P9c — 把 PNN 放大到 resnet18 + 結合 softmax-KD**：如 P8b→P8c 量容量效應；或 PNN + softmax-KD 混合逼近 buffered 上界。機制已確立，優先序中等。
- **（可選）P8e — α / buffer-size / lr 掃描與更長 stream**：刻畫甜蜜點邊界與正向遷移上限。機制已確立，優先序中等。
- **（L4 方向，遠程）task-free + 開放世界 + 漂移 + 新奇偵測/容量增長**：P3 已在 numpy 端證明 task 邊界幾乎可免費移除，但 P8 的 backbone 版尚未驗證 task-free；開放世界/漂移/自主長容量完全未碰，這才是「真正的持續學習」(L4) 與 LLM 終身學習接軌處。

### ✅ P1 — 攻 Class-IL 的遺忘缺口（已完成，§11.3）
**結果**：`NCMReplayEWC`（最近類別原型讀出，iCaRL 式）把 Class-IL final 0.31→**0.858**、遺忘 0.50→**0.06**、retention>1.0（3 seeds, std 0.003）。診斷正確：病灶是線性頭的 recency/magnitude bias，換成無偏原型讀出即解。候選清單裡的 cosine/BiC/class-balanced replay 尚未試（NCM 已夠強，這些可作為進一步小幅優化或在更大規模時備用）。

### P2 — Conflicting-task benchmark（任務真正衝突）+ 統一評分標準（當前工作）
- **為什麼**：目前所有任務骨子裡共享同一函數（sum→bucket），head/adapter 吸收差異，使「不遺忘」異常容易（retention→1.0）。真實 CL 的任務彼此衝突：學 B 會主動破壞 A 的共享層解。這是檢驗目前好結果是不是「benchmark 太友善」造成的關鍵。
- **做法**：在 `benchmark.py` 加一個 mode，讓每個 task 的底層 input→class 函數**真的不同**，目前採用 `sum / weighted_sum / first_half_sum / second_half_sum / adjacent_product_sum` 輪換，使共享層面臨衝突的特徵需求。先保持多頭 Task-IL（隔離 head，純測共享層衝突），後續再考慮 Class-IL/adapter 版本。
- **統一評分**：同時建立一個簡單 scorecard，不只看 final accuracy，而是一起看：
  - final / Joint upper bound 的比例（不同 benchmark 才可比較）。
  - retention ratio、BWT、mean forgetting。
  - 是否需要 task id、task boundary、raw replay buffer。
  - 計算/記憶限制：buffer size、是否存 logits/prototypes、是否存原始輸入。
- **驗收**：在新 stream 上跑 `Naive`、`ReplayEWC`、`DarkReplayEWC --dark-alpha 0.5`、`HippocampalReplayEWC`、`Joint`。若 retention 從 ~1.0 明顯下降，就找到了「友善 benchmark」的邊界；若 DER++ 仍穩，才更像真正可持續學習骨幹。
- **進度（2026-06-27）**：已新增 `conflicting` mode，規則輪換 `sum / weighted_sum / first_half_sum / second_half_sum / adjacent_product_sum`，並接上 `run.py` / `analyze.py` / smoke test。20 tasks × 1000 steps × 2 seeds sanity：
  - Joint upper-bound 約 **46%**，代表任務可學但明顯比原始 Task-IL 困難。
  - ReplayEWC 約 **39%**，Naive 約 **26%**。
  - DarkReplayEWC（DER++ α=0.5）約 **30%**，初步顯示「函數錨定」在真衝突任務可能過度保守，不能直接沿用友善 benchmark 的結論。
  - HippocampalReplayEWC 約 **13%**，episodic prototype readout 在此設定明顯不適配。
  這只是 sanity，不是正式結果；下一步要跑較長步數/更多 seeds，並用 final/Joint ratio + forgetting scorecard 判讀。
- **校準結果（20 tasks × 3000 steps × 3 seeds）**：
  - Joint upper-bound：**0.592 ± 0.007**。
  - ReplayEWC：**0.504 ± 0.007**，final/Joint **0.852**，BWT -0.005，forget 0.083，retention 0.992。
  - DarkReplayEWC（DER++ α=0.5）：**0.456 ± 0.004**，final/Joint **0.771**，BWT -0.035，forget 0.064，retention 0.932。
  初步結論：conflicting benchmark 可學但明顯更硬；DER++ 在原始 Task-IL 長流是英雄，但在底層函數真衝突時會拖累 final/Joint ratio。這支持「沒有單一全勝機制，應依任務衝突程度調整函數錨定強度」。
- **40-task scorecard（3000 steps × 3 seeds，已寫入 `report.md` §12）**：
  - Joint：**0.643 ± 0.007**。
  - Naive：0.245 ± 0.032，final/Joint 0.381，forget 0.333。
  - **ReplayEWC：0.486 ± 0.001，final/Joint 0.756，BWT -0.054，forget 0.112，retention 0.902。**
  - DarkReplayEWC α=0.1：0.463 ± 0.003，final/Joint 0.719，forget 0.102。
  - DarkReplayEWC α=0.25：0.458 ± 0.004，final/Joint 0.711，forget 0.093。
  - DarkReplayEWC α=0.5：0.449 ± 0.006，final/Joint 0.698，forget 0.087。
  結論：α 越大，forgetting 越低，但 final/Joint 越差。DER++ 在「共享底層函數」時是英雄，在「底層函數真衝突」時會過度錨定。下一步應做 **adaptive distillation strength / conflict detector**，而不是固定開 DER++。

### ✅ P2.5 — Pressure / maturity gating（已完成，§12.3）
**做完的事情**：
- 新增 `PressureDarkReplayEWC`：可靠記憶（stored logits 高信心且正確）× label-loss forgetting pressure。
- 將 `DarkReplayEWC` 加上 delayed distillation/maturity gate：`--distill-start-task`、`--distill-ramp-tasks`。
- 補上 pressure telemetry：reliability、loss pressure、drift pressure、effective alpha。
- 跑完 20-task sanity、80-task pressure 對照、130-task ReplayEWC/DarkReplayEWC/Pressure/delayed-DER++ 對照。

**做完後的結果**：
- 20-task sanity（loss-only default）：conflicting **0.382 ± 0.010**、label_permuted **0.598 ± 0.005**，幾乎退回 ReplayEWC，可避開固定 DER++ 短流傷害。
- 130-task label_permuted：ReplayEWC **0.815 ± 0.074**；full DarkReplayEWC α=0.5 **0.892 ± 0.019**；PressureDarkReplayEWC **0.847 ± 0.030**；delayed DER++ start=40 **0.827 ± 0.024**；start=80 **0.779 ± 0.047**。
- 結論：reactive pressure 能降低壞 seed 風險但太保守；maturity gate 太粗；logit drift 是假警報；full DER++ 的優勢是 proactive consolidation。
- 下一步不應再調單一 alpha，而是做 **P2.6：自動辨識「共享規則長流」vs「真衝突/短流」regime**。

### ✅/❌ P2.6 — cos-RTP regime detector（已做完並 3-seed 驗收，結論為負面，§12.4）
**做完的事情**：實作 `LookaheadDarkReplayEWC`（單步反事實，因雜訊放棄）與 `RtpDarkReplayEWC`（task-onset 前 5 步累積共享層梯度 vs buffer 歷史梯度餘弦，>= threshold 判 synergistic→α=0.5、否則 conflicting→α=0）。CLI：`--rtp-cos-threshold`、`--rtp-probe-steps`。

**3-seed 驗收結果（推翻初版 1-seed「成功」）**：
- 130-task label_permuted：RTP **0.818 ± 0.042**，未達 ≥0.87 目標、略低於 ReplayEWC **0.832**，遠不及 full DER++ **0.893**。
- 40-task conflicting：RTP 0.487（f/J 0.646）≈ ReplayEWC 0.486（0.644），達標但只因退回 ReplayEWC。
- 20/80-task label_permuted：RTP 0.865 / 0.816 ≈ ReplayEWC 0.873 / 0.818，達標但只因退回 ReplayEWC。
- 診斷：RTP `alpha_mean≈0.011`（全程僅 ~2% task 判 synergistic）= 幾乎永遠關閉 DER++。threshold 掃描：-0.05→0.818、-0.2→0.810、**-1.0（強制永遠開）→0.898≈DER++**。

**結論**：機制完好，**task-onset 權重梯度餘弦不是有效的 regime 訊號**（多頭標籤排列把不同反傳誤差旋進共享層，污染梯度方向，使共享規則長流與真衝突都被判 conflicting）。P2.6 目標未達成，退回 backlog 成 P2.7。結果檔：`results_rtp_*.json`。

### ✅/⚠️ P2.7 — Horizon oracle regime detector（已完成，§12.5，部分正面）
**做完的事情**：實作 `HorizonDarkReplayEWC`，用已知/覆寫的 stream horizon 做 oracle gate：
- 若 `horizon >= horizon_threshold`（預設 80），從一開始啟用 DER++，等價 `DarkReplayEWC α=0.5`。
- 若 `horizon < horizon_threshold`，`_distill_age_scale=0`，等價 `ReplayEWC`。
- CLI：`--horizon-threshold`、`--horizon-override`。`run.py` 也會記錄 `horizon/horizon_threshold/horizon_regime` diagnostics。

**標準驗收結果（3 seeds）**：
- label_permuted 20-task × 4000：**0.873 ± 0.015**，`horizon_regime=short`、`alpha_mean=0`，貼近 ReplayEWC（短流保守）。
- label_permuted 80-task × 4000：**0.868 ± 0.019**，`horizon_regime=long`、`alpha_mean=0.5`，等價 full DER++，高於 ReplayEWC 0.818。
- label_permuted 130-task × 4000：**0.893 ± 0.019**，等價 full DER++，達成 ≥0.87 目標。
- conflicting 40-task × 3000：**0.486 ± 0.002**，`alpha_mean=0`，等價 ReplayEWC，避開 DER++ 在真衝突時的傷害。

**限制 / 新教訓**：
- 這不是完整 detector，而是 oracle validation。它證明「可切換 schedule」存在：同一套 trainer 可以同時拿到短流/衝突安全性與長流 DER++ 增益。
- 但 **horizon 長度本身不夠**：80-task × 2000 steps 下，Horizon/DarkReplayEWC **0.736** < ReplayEWC **0.788**。也就是「任務數長」還要搭配足夠訓練 budget / replay 品質，DER++ 才值得開。
- 因此 P2.7 把問題縮小成：不要再問「能不能切換」，而是問「如何在線上估計 DER++ 的函數空間收益」。

### ✅/⚠️ P2.8 — online function-space benefit detector（已完成，§12.6，部分正面/負面）
**做完的事情**：新增 `BenefitDarkReplayEWC`。它不偷看 `n_tasks`，而是在固定步距做兩個可回復虛擬更新：`alpha=0`（ReplayEWC）與 `alpha=dark_alpha`（DER++）。比較舊 replay label loss / stored-logit MSE 的改善，以及當前 batch loss 的傷害，再用 EMA 控制 alpha。CLI：`--benefit-probe-interval`、`--benefit-ema-decay`、`--benefit-threshold`、`--benefit-alpha-lr`、`--benefit-harm-weight`、`--benefit-logit-weight`、`--benefit-min-old`。

**3-seed 結果（label-loss benefit 預設，`benefit_logit_weight=0`）**：
- label_permuted 20×4000：**0.873 ± 0.015**，alpha_mean≈0，等於 ReplayEWC，短流安全。
- label_permuted 80×2000：**0.788 ± 0.030**，alpha_mean≈0，修掉 P2.7 horizon oracle 的誤開（Horizon/Dark 0.736）。
- conflicting 40×3000：**0.486 ± 0.002**，alpha_mean≈0，等於 ReplayEWC，真衝突安全。
- label_permuted 80×4000：**0.818 ± 0.045**，alpha_mean≈0，未拿到 full DER++ 0.868。
- label_permuted 130×4000：**0.832 ± 0.079**，alpha_mean≈0，未達 ≥0.87，等於 ReplayEWC、低於 full DER++ 0.893。

**ablation / 教訓**：
- `benefit_logit_weight=0.25` 會把 stored-logit MSE 改善當收益，因此 alpha 幾乎全開；但短流 label 20 掉到 **0.482**（ReplayEWC 0.602）、conflicting 20 掉到 **0.317**（ReplayEWC 0.384）、80×2000 seed0 掉到 **0.732**（ReplayEWC seed0 0.820）。
- label-loss probe 是安全的，但太短視：它看到 DER++ 一步後讓 replay label loss / current loss 變差，所以永遠關閉；然而 full DER++ 的長流收益來自多步、慢時間尺度的 proactive consolidation。
- logit-MSE probe 會看到「舊函數幾何更像」，但這正是 P2.5/P2.8 反覆證明的假陽性：保幾何不等於保有用任務表現。

### ✅/❌ P2.9 — local multi-step / slow-benefit controller（已完成，§12.7，負面結果）
**做完的事情**：新增 `SlowBenefitDarkReplayEWC`。它把 P2.8 的一步虛擬更新延長成最近 `slow_rollout_steps=5` 個 current batches 的可回復 shadow rollout，仍比較 `alpha=0` vs `alpha=dark_alpha`，並沿用 P2.8 的 benefit score / EMA。CLI：`--slow-rollout-steps`；diagnostics：`slow_rollout_steps`、`benefit_*`、`dark_effective_alpha_*`。

**3-seed 結果（recent-window rollout，label-loss benefit，`benefit_logit_weight=0`）**：
- label_permuted 20×4000：**0.873 ± 0.015**，alpha=0，短流安全。
- label_permuted 80×2000：**0.788 ± 0.030**，alpha=0，保持 P2.8 安全性。
- conflicting 40×3000：**0.486 ± 0.002**，alpha=0，真衝突安全。
- label_permuted 80×4000：**0.818 ± 0.045**，alpha=0，未拿到 full DER++ 0.868。
- label_permuted 130×4000：**0.832 ± 0.079**，alpha=0，未達 ≥0.87，低於 full DER++ 0.893。

**ablation / 教訓**：
- `benefit_logit_weight=0.02`（seed0）在 conflicting 40 尚安全（0.486），但 80×2000 掉到 **0.771**（ReplayEWC/P2.8 seed0 0.820），130 也只有 **0.866**（P2.8 seed0 0.890）。小 logit 權重仍是假陽性。
- local 5-step window 比 one-step 更長，但仍只看到 DER++ 的短期 label/current harm，看不到從 stream 早期開始維持 soft function geometry 的長期複利。
- P2.9 排除的是「同一局部視窗多走幾步」；真正未試的是 persistent shadow-model bandit：shadow_on/off 必須從早期就共同經歷真實 stream，而不是每次 probe 才從當前模型複製。

### ✅ P3 — Task-free（無邊界）CL（已完成，§13）
**做完的事情**：新增 `OnlineEWCReplay` / `OnlineDarkReplayEWC`（`trainers.py`）。`on_task_end` 改 no-op；Fisher/anchor 改由 `_online_consolidate_fisher` 每 `consolidate_every` 步從 reservoir buffer 取樣估計（多頭按 head 分組前傳），squared-grad 累加與 `fisher_decay` EMA 與邊界版一致，`lam` 可沿用。CLI：`--consolidate-every`、`--fisher-sample`。smoke test 三 mode 全過。

**結果（80-task label_permuted × 3 seeds，consolidate_every=400）**：
- ReplayEWC（邊界）0.818 → **OnlineEWCReplay（無邊界）0.826**（+0.008，遺忘更低）。
- DarkReplayEWC（邊界）0.868 → **OnlineDarkReplayEWC（無邊界）0.855**（−0.012，在 std ±0.015 內）。
- 無邊界 DER++ 仍勝過有邊界 ReplayEWC。**失去邊界知識的代價趨近於零。**
- 結果檔：`results_taskfree_label_permuted.json`、`results_taskfree_misaligned137_label_permuted.json`（故意把 consolidate 步距與 task 長度錯位的穩健性檢驗）。

**結論**：reservoir replay 才是主力抗遺忘機制（本來就無邊界），Fisher 鞏固只需粗略滾動估計，精確 task-end 時機可有可無——核心配方天生接近 task-free。注意：多頭 Task-IL 的 head（與 input_adapter 的 adapter）在 train/eval 仍需 task id 做路由，這是架構需求、不是鞏固時機的邊界知識；P3 移除的是後者。若要連 head 路由都 task-free，須走 single-head class_il（見 §11）或 task-inference 機制。

### ✅ P4 — Buffer-free / generative replay（已完成，§14）
**做完的事情**：新增 `GenerativeReplayEWC`（`trainers.py`）——完全不存原始樣本，改對每 (task,class) 維護因子化 categorical 生成模型（per-position 數字頻率充分統計量），回放時抽合成 one-hot 窗口、經對應 head 路由（重用既有多頭/adapter 路由）。Fisher 仍在 `on_task_end` 用當前任務瞬時資料。關鍵設計 **sum-matched conditional generation**：rejection-sample 讓合成窗口的數字和落在類別和分布 `mean±gen_sum_tol·std` 內（因為標籤由和定義）。CLI：`--gen-smoothing`、`--gen-sum-tol`、`--gen-sum-match-off`（ablation）。smoke 三 mode 全過。

**結果**：
- sum-match ablation（10-task×1 seed）：關掉 0.470 → 打開 **0.753**（diag 0.569→0.688、forget 0.138→0.040）。
- label_permuted 交叉：10-task raw 0.810 > gen 0.753（buffer 餵得飽）；**80-task gen 0.916 > raw ReplayEWC 0.818 > raw DER++ 0.868**（3 seeds；固定 buffer 被稀釋、生成統計量不衰減；gen 方差更小、遺忘更低）。
- conflicting 40-task：gen **0.431** < raw ReplayEWC 0.486（sum-matched 只條件在總和，對 weighted/half/product 規則是錯統計量）。
- 結果檔：`results_genreplay_{label_permuted,longstream_label_permuted,conflicting_40,10task_label_permuted,nomatch_ablation_10task}.json`。

**結論**：不存原始樣本可行、且在「條件統計量對得上 + 串流夠長」時甚至超越 raw replay；瓶頸是生成模型能否抓住定義標籤的統計量，而非記憶機制本身。→ 開 P5。

### ✅ P5 — rule-agnostic 生成式回放（已完成，§15）
**做完的事情**：兩條路。
- `NBGenerativeReplayEWC`（候選 a 的廉價版：用儲存 categorical 自建 naive-Bayes 分類器做 rejection）——**失敗**，conflicting 40-task 0.418 < sum-match 0.431，label_permuted 80-task 退步到 0.521（forget 0.256）。因子化邊際做的分類器太弱，accept region 無效。
- `ScholarGenerativeReplayEWC`（候選 c：每任務 `on_task_end` 把模型 `copy.deepcopy` 成凍結 teacher；replay 時抽合成輸入、用 teacher soft logits 蒸餾，generative DER++，只蒸餾 `task ≤ scholar_max_task` 的舊任務）——**成功**。CLI：`--dark-alpha`、`--replay-weight`。
  - conflicting 40-task：**0.474（final/Joint 0.628 vs raw 0.644，差 1.6%≤2%）**、forget **0.082**（最低）。
  - label_permuted 80-task：0.809 ≈ raw 0.818，但不及 sum-match 峰值 0.916。
- 結果檔：`results_p5_{nb,scholar}_{conflicting_40,label_permuted}.json`。

**結論**：沒有單一 buffer-free 生成器全勝——已知簡單統計量用 sum-match（甚至贏 raw），規則複雜/未知用 scholar（一份常數快照換 rule-agnostic 標註）。NB 證實「從儲存統計量自建分類器」太弱。殘留缺口（conflicting 仍差 raw 一點、scholar 拿不到 label 峰值）都源自因子化輸入保真度 → P6。

### ✅/❌ P6 — 更強的輸入生成器（已完成，§16，結論為負面）
**做完的事情**：實作 `ScholarGlobalGenerativeReplayEWC`——驗證 P5 的「輸入保真度（離流形）」歸因。觀察 pi 數位 ~iid uniform→真實輸入分布本就因子化，P5 從**每類別**邊際抽樣才是偏斜離流形；改從**全域 per-task 邊際**抽樣（≈ 真實 uniform＝on-manifold）+ scholar 標註。

**結果（3 seeds）推翻 on-manifold 假設**：
- conflicting 40-task：scholar-global **0.428 ± 0.011** < scholar-class 0.474 < raw 0.486，forget 最低 0.052。
- label_permuted 80-task：scholar-global **0.769 ± 0.027** < scholar-class 0.809 < sum-match 0.916。
- 瓶頸是**可塑性**（diag 0.54→0.45）：全域均勻蒸餾過度正則共享層、稀釋類界訊號。**per-class 集中回放 > 全域 on-manifold 覆蓋。**
- 結果檔：`results_p6_scholar_global_{conflicting_40,label_permuted}.json`。

**結論**：殘留缺口不是 on-manifold 與否的問題。scholar-class 距 raw 的 1.6% 小差距更像**真實樣本不可取代的價值**（精確 per-class 聯合結構→正向後向遷移，raw conflicting 有 +BWT、合成回放沒有）。P4–P6 總結：不存原始樣本可行且常足夠（長流甚至贏 raw），完全追平 raw 仍有一道由真實樣本聯合結構撐起的小硬牆。唯一未試：per-class autoregressive（見上方 Todo P6b，優先序低）。

## 3. 建議順序與理由（P1–P8、P8b/c/d、P9、P10 已完成；P2.6/P2.9/P9 負面；P2.10/P6b 已停損）

1. **P9b（buffer-free 累積的其他機制）** — P9 證明純蒸餾(LwF)不行；最有潛力的是把生成式回放(P4–P6)搬到會動 backbone 補回「真實樣本」角色。這是把 L3 累積做成 buffer-free 的核心開放問題。
2. **（L4，遠程）task-free + 開放世界 + 漂移** — 真正的持續學習，與 LLM 終身學習接軌；目前基本未碰。
3. **（可選）P8e（α/buffer/stream 掃描）** — 刻畫甜蜜點邊界。機制已確立，優先序中等。
- ~~P2.10（自動 DER++ gating）~~ / ~~P6b（per-class autoregressive）~~ — **已停損**。

## 4. 慣例

- 每個實驗存成 `results_<描述>.json`；多 seed（≥3）。
- 新 trainer 加進 `trainers.py` 的 `TRAINER_REGISTRY`，`test_smoke` 會自動覆蓋。
- 跑完更新 `report.md`（新增小節 + 更新 §0 TL;DR 與結論）與本檔（移除已完成項）。
- 誠實第一：負結果（像 GPM、DER++ 在 Class-IL 反轉）和正結果一樣要記錄。
- 有結果，也要把結論更新到 report.md，不要只改 NEXT_STEPS.md，否則會斷層
- 結束一輪改善與測試結果，做一次 git commit 。
