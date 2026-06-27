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
- **正向遷移**：表徵層有（晚段任務最終準確率更高），學習速度沒有（§10.3）。

## 2. Backlog（依優先序；每項含 為什麼 / 做法 / 驗收）

### 工作狀態總覽（先看這裡）

**已完成 / 做完的事情**
- **P1 — Class-IL 遺忘缺口**：已用 `NCMReplayEWC` 解掉線性頭 recency bias，詳見 §11.3。
- **P2 — Conflicting-task benchmark**：已新增真衝突任務、40-task scorecard、DER++ alpha sweep，詳見 §12.1。
- **P2.5 — Pressure / maturity gating**：已實作 `PressureDarkReplayEWC`、confidence/loss-pressure gate、delayed DER++ maturity gate，並完成 20/80/130-task 對照，詳見 §12.3。

**下一個要做 / Todo**
- **P2.6 regime / horizon detector**：不要再調單一 alpha，而是讓系統判斷目前 stream 屬於「共享規則長流」還是「真衝突/短流」，再決定 `distill_mode = off / pressure / full`。
- P2.6 完成後再進 P3（task-free）與 P4（buffer-free / generative replay）。

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
- **adaptive/confidence-gated distillation sanity（20 tasks × 1000 steps × 2 seeds，已寫入 `report.md` §12.2）**：
  - conflicting：ReplayEWC **0.384 ± 0.008**；固定 DarkReplayEWC α=0.5 **0.305 ± 0.002**；`AdaptiveDarkReplayEWC` **0.384 ± 0.008**（α 平均降到 0）；confidence-gated DarkReplayEWC **0.371 ± 0.007**。
  - label_permuted：ReplayEWC **0.602 ± 0.004**；固定 DarkReplayEWC α=0.5 **0.470 ± 0.005**；`AdaptiveDarkReplayEWC` **0.601 ± 0.004**；confidence-gated DarkReplayEWC **0.550 ± 0.011**。
  - 結論：gradient-conflict gate 是有效安全閥，可避免 DER++ 在真衝突下傷害模型；confidence gate 支持「可靠記憶才鞏固」的假設。但 current-vs-dark 梯度 cosine 在短流 label_permuted 也偏負，不能當完整 regime detector。下一步不要只調 α，應找 **何時開蒸餾** 的訊號（長期遺忘壓力、任務相似度、reliability × drift policy）。

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

### P2.6 — Regime / horizon detector（下一個最有價值工作）
**要做的事情**：不要再調單一 alpha，而是讓系統判斷目前 stream 屬於「共享規則長流」還是「真衝突/短流」，再決定 `distill_mode = off / pressure / full`。

- **為什麼**：目前已知道 full DER++ 在共享規則長流最強，但在真衝突/短流有害；ReplayEWC/Pressure 比較安全但拿不到 full DER++ 的長流增益。局部訊號（gradient cosine、logit drift、label-loss pressure、單純 task count）都不夠。需要一個更上層的 policy 判斷「這條 stream 是否值得 proactive consolidation」。
- **做法候選**：
  - **雙軌 shadow probe**：主模型用 ReplayEWC；低成本 shadow 指標估計「若開 DER++，replay loss/old accuracy 是否改善且 current-task diagonal 是否不受傷」。可先不維護完整第二模型，只在 replay batch 上計算反事實 gradient/短步 lookahead。
  - **regime scorecard online 化**：維護最近窗口的 old-task label loss、current-task learning slope、replay/current gradient conflict、stored-logit reliability、任務數/horizon；用簡單規則輸出 `distill_mode ∈ {off, pressure, full}`。
  - **先用 oracle validation 做上界**：離線掃描每個任務區段應開/關 DER++ 的 schedule，確認「可學 schedule」真的存在，再做線上 detector。
- **驗收**：在三個設定同時測：
  - label_permuted 130-task：接近 full DER++（目標 ≥0.87）。
  - conflicting 40-task：接近 ReplayEWC（目標 final/Joint 不低於 ReplayEWC 2%）。
  - label_permuted 20/80-task：不能明顯低於 ReplayEWC。

### P3 — Task-free（無邊界）CL
- **為什麼**：目前都靠 `on_task_end`（算 Fisher、更新 adapter/GPM 基、Class-IL 切片）——等於知道任務何時切換。真實串流沒有邊界。
- **做法**：拿掉 `on_task_end` 依賴：純 reservoir replay 本來就 boundary-agnostic；EWC 改成線上 Fisher（每步用 running estimate）或改用 replay-only。可選加上 online 漂移偵測。
- **驗收**：對照「給邊界 vs 不給邊界」的 ReplayEWC/DER++，量化失去邊界知識的代價。

### P4 — Buffer-free / generative replay
- **為什麼**：DER++ 仍存 2000 筆原始樣本；真正的終身學習可能不准存原始資料、串流無上限。這一步應在 P2/P3 之後做，因為先要知道方法在「真衝突」和「無邊界」下是否仍成立。
- **做法**：實作不存原始輸入的版本：
  - **feature replay**：只存/重播倒數第二層特徵（a2）而非原始 X；或
  - **小型 generative replay**：每類別一個 Gaussian（mean+cov）在輸入或特徵空間合成舊資料；
  - 對照原始 buffer 的 DER++/ReplayEWC。
- **驗收**：在 label_permuted 130-task 與 P2 conflicting-task 上，buffer-free 版本 retention 能不能接近原始 DER++。

## 3. 建議順序與理由（P1 已完成）

1. **P2.6（regime / horizon detector）** — 目前最高價值：P2/P2.5 已證明「full DER++ 很強但只適合共享規則長流；ReplayEWC/Pressure 安全但較保守」。下一步要讓系統自動選 `off / pressure / full`，而不是再人工調 α。
2. **P3（task-free）** — 若 P2.6 能在有 task boundary 的 setting 自動選對 regime，下一個硬限制是沒有任務邊界；這會直接挑戰 EWC anchor、Fisher、prototype 更新等目前依賴 `on_task_end` 的機制。
3. **P4（buffer-free / generative replay）** — 最後再拿掉 raw replay buffer。這是更接近真實終身學習的記憶限制，但應在確認方法能通過真衝突、regime selection 與無邊界後再做。

## 4. 慣例

- 每個實驗存成 `results_<描述>.json`；多 seed（≥3）。
- 新 trainer 加進 `trainers.py` 的 `TRAINER_REGISTRY`，`test_smoke` 會自動覆蓋。
- 跑完更新 `report.md`（新增小節 + 更新 §0 TL;DR 與結論）與本檔（移除已完成項）。
- 誠實第一：負結果（像 GPM、DER++ 在 Class-IL 反轉）和正結果一樣要記錄。
- 有結果，也要把結論更新到 report.md，不要只改 NEXT_STEPS.md，否則會斷層
- 結束一輪改善與測試結果，做一次 git commit 。
