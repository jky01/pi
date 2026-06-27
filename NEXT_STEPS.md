# NEXT_STEPS.md — 未完成工作與後續計畫（給下次運行參考）

這份檔案記錄「還沒做的工作項目 + 為什麼做 + 預計怎麼做」，讓下次運行（cold start）能直接接續，不必重新推導已有結論。完成一項就把它從 backlog 移到「已完成」並更新 `report.md`。

---

## 0. 環境與如何跑（重要，先讀）

- **Python 直譯器**：系統預設的 `python` / `python3` **沒有 numpy**。要用這支（有 numpy 2.3.5）：
  `/Users/jackyyeh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`
- 跑 benchmark：`$PY run.py <mode> --methods ... --seeds 0 1 2 --n-tasks N --steps-per-task S --output results_xxx.json`
  - mode ∈ `label_permuted`（Task-IL 多頭）、`input_permuted`（Domain-IL 單頭，加 `--input-adapter`）、`class_il`（單頭、無 task id）。
- 冒煙測試（目前 registry 內所有 trainers × label/input/conflicting 三種 mode）：`$PY -m unittest test_smoke`
- 長串流受 `pi_digits_600000.txt` 600k 位數限制：`n_tasks*(steps+test) ≲ 560k`。
- 結果分析：`$PY analyze.py <mode>`，或直接讀 `results_*.json`（`run.summarize` 算 final/bwt/forget/retention）。

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

- **P3 — Task-free (無邊界) CL（已完成，§13）**：新增 `OnlineEWCReplay` / `OnlineDarkReplayEWC`，把 Fisher/anchor 鞏固從 `on_task_end` 邊界觸發改成固定步距線上估計（從 reservoir buffer 取樣）。80-task × 3 seeds：失去邊界知識的代價趨近於零（EWC +0.008、DER++ −0.012 在 std 內），無邊界 DER++ 仍勝過有邊界 ReplayEWC。
- **P4 — Buffer-free / generative replay（已完成，§14）**：新增 `GenerativeReplayEWC`（完全不存原始樣本，per-(task,class) categorical 生成模型 + sum-matched conditional generation）。label_permuted 80-task buffer-free **0.916 反超** raw ReplayEWC 0.818 與 DER++ 0.868（長流 buffer 被稀釋、生成統計量不衰減）；conflicting 0.431 < raw 0.486（sum-matched 條件統計量對非 sum 規則不符）。
- **P5 — rule-agnostic 生成回放（已完成，§15）**：試兩條路。`NBGenerativeReplayEWC`（從儲存 categorical 自建 NB 分類器 rejection）**失敗**（conflicting 0.418、label 退步到 0.521）。`ScholarGenerativeReplayEWC`（每任務凍結 teacher、對合成輸入 soft-logit 蒸餾，generative DER++）**成功補上 conflicting 缺口**：0.474（final/Joint 0.628 vs raw 0.644，差 1.6%≤2%）、遺忘最低 0.082；label_permuted ≈ raw（0.809）但不及 sum-match 峰值 0.916。沒有單一 buffer-free 生成器全勝；殘留缺口源自因子化輸入保真度 → P6。
- **P6 — 更強的輸入生成器（已完成，§16，負面結果）**：`ScholarGlobalGenerativeReplayEWC`（全域 per-task 邊際抽 on-manifold 輸入 + scholar 標註）**推翻 P5 的 on-manifold 歸因**——3-seed 全面更差（conflicting 0.428 < scholar-class 0.474；label 0.769 < 0.809），儘管 forgetting 最低（0.052），瓶頸是可塑性（diag 0.54→0.45）。per-class 集中回放比全域覆蓋更重要；殘留小差距更像真實樣本不可取代的價值（精確 per-class 聯合結構→正向後向遷移），非輸入分布失配。

**下一個要做 / Todo**
- **P2.7 — function-space / horizon regime 偵測器（接續 P2.6 的未竟目標）**：見下方 backlog。task-onset 權重梯度餘弦已證實無效，改用反事實 replay-accuracy 量測或 horizon 訊號。
- **（可選）P6b — per-class autoregressive 生成器**：P6 證明「全域 on-manifold」方向錯；唯一還沒試的是 per-class **autoregressive**（抓類內位置相關）能否把 scholar-class 從 0.474 再推近 raw 0.486。但邊際空間極小（僅差 1.6% final/Joint），優先序低。

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

### P2.7 — function-space / horizon regime detector（接續 P2.6 未竟目標）
**要做的事情**：讓系統判斷目前 stream 是「共享規則長流（該開 DER++）」還是「真衝突/短流（該用 ReplayEWC）」，但**不要再用 task-onset 權重梯度餘弦**（P2.6 已證實無效），也不要再用 logit drift / label-loss（P2.3/P2.5 已證實是 reactive、假警報多）。

- **為什麼**：full DER++ 在共享規則長流最強（130-task 0.893）但在真衝突/短流有害；ReplayEWC 安全但拿不到長流增益。需要一個能在線上、用**函數空間後果**判斷「開 DER++ 是否真的有益」的訊號。
- **做法候選（依 P2.6 教訓更新）**：
  - **反事實 replay-accuracy probe（首選）**：週期性做小型反事實量測——在一小批 replay 上比較「開 DER++ 蒸餾 vs 不開」對**舊任務 replay accuracy** 的影響，以及對**當前任務 diagonal** 的傷害。若舊任務有改善且新任務不受傷→開；否則→關。這是 function-space 後果量測，繞過 P2.6 的權重方向污染。
  - **oracle validation 先做上界**：離線掃描「每段任務該開/關 DER++」的 schedule，確認存在「可學 schedule」能同時逼近 130-task DER++ 與 40-task ReplayEWC，再做線上 detector。若 oracle 都做不到，代表單一可切換 α 不夠，需要更細的 per-layer / per-task 蒸餾。
  - **horizon 訊號**：估計剩餘 stream 長度 / 任務重複度；長流才值得 proactive consolidation。
- **驗收**（同 P2.6，未變）：
  - label_permuted 130-task：接近 full DER++（目標 ≥0.87）。
  - conflicting 40-task：接近 ReplayEWC（目標 final/Joint 不低於 ReplayEWC 2%）。
  - label_permuted 20/80-task：不能明顯低於 ReplayEWC。

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

## 3. 建議順序與理由（P1/P2/P2.5/P3/P4/P5/P6 已完成；P2.6 做完但結論為負面）

1. **P2.7（function-space / horizon regime detector）** — 接續 P2.6 未竟目標（自動切換 DER++），但 P2.6 已證實 task-onset 權重梯度餘弦無效，要改用反事實 replay-accuracy probe，風險較高；建議**先用 oracle validation 確認可學 schedule 存在**再投入。這是目前 backlog 中最有開放價值的題目。
2. **（可選）P6b（per-class autoregressive 生成器）** — 邊際空間極小（scholar-class 距 raw 僅 1.6%），優先序低。

## 4. 慣例

- 每個實驗存成 `results_<描述>.json`；多 seed（≥3）。
- 新 trainer 加進 `trainers.py` 的 `TRAINER_REGISTRY`，`test_smoke` 會自動覆蓋。
- 跑完更新 `report.md`（新增小節 + 更新 §0 TL;DR 與結論）與本檔（移除已完成項）。
- 誠實第一：負結果（像 GPM、DER++ 在 Class-IL 反轉）和正結果一樣要記錄。
- 有結果，也要把結論更新到 report.md，不要只改 NEXT_STEPS.md，否則會斷層
- 結束一輪改善與測試結果，做一次 git commit 。
