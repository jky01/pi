# NEXT_STEPS.md — 未完成工作與後續計畫（給下次運行參考）

這份檔案記錄「還沒做的工作項目 + 為什麼做 + 預計怎麼做」，讓下次運行（cold start）能直接接續，不必重新推導已有結論。完成一項就把它從 backlog 移到「已完成」並更新 `report.md`。

---

## 0. 環境與如何跑（重要，先讀）

- **Python 直譯器**：系統預設的 `python` / `python3` **沒有 numpy**。要用這支（有 numpy 2.3.5）：
  `/Users/jackyyeh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`
- 跑 benchmark：`$PY run.py <mode> --methods ... --seeds 0 1 2 --n-tasks N --steps-per-task S --output results_xxx.json`
  - mode ∈ `label_permuted`（Task-IL 多頭）、`input_permuted`（Domain-IL 單頭，加 `--input-adapter`）、`class_il`（單頭、無 task id）。
- 冒煙測試（16 trainers × 2 mode）：`$PY -m unittest test_smoke`
- 長串流受 `pi_digits_600000.txt` 600k 位數限制：`n_tasks*(steps+test) ≲ 560k`。
- 結果分析：`$PY analyze.py <mode>`，或直接讀 `results_*.json`（`run.summarize` 算 final/bwt/forget/retention）。

## 1. 已確立的結論（不要重做，直接引用 report.md）

- **瓶頸是遺忘、不是可塑性流失**：有 replay 時對角線到 250 tasks 仍上升（§9）。
- **Task-IL 最佳配置**：`Replay + Fisher-EWC + DER++ logit 蒸餾`（= `DarkReplayEWC --dark-alpha 0.5`），130/250-task retention ≈ 1.0、BWT≈0（§10）。
- **input_permuted 需要 per-task 輸入 adapter**（`--input-adapter`），12%→55%（上界 81%，§7）。
- **Benna-Fusi / SustainableReplayEWC**：機制有效但只在 **replay-free** regime 才有價值；有 replay 時被 Fisher-EWC 支配（§9.4）。
- **GPM 梯度投影不適配本 benchmark**（輸入平穩→凍結共享層）；函數空間錨定要用「輸出蒸餾」不是「輸入子空間投影」（§10.1）。
- **Class-IL（誠實硬測試，§11）**：拿掉 task id 後線性頭因 recency bias 崩壞（**DER++ 反轉成有害**）。**已解：`NCMReplayEWC`（最近類別原型讀出，iCaRL 式）把 0.31→0.858、遺忘 0.50→0.06、retention>1.0**（§11.3）。教訓：表徵骨幹用 replay+EWC，讀出依設定換（Task-IL：head+DER++；Class-IL：無偏原型）。benchmark Class-IL 上限 ~200 類（K=8 位數和僅 ~73 相異值）。
- **正向遷移**：表徵層有（晚段任務最終準確率更高），學習速度沒有（§10.3）。

## 2. Backlog（依優先序；每項含 為什麼 / 做法 / 驗收）

### ✅ P1 — 攻 Class-IL 的遺忘缺口（已完成，§11.3）
**結果**：`NCMReplayEWC`（最近類別原型讀出，iCaRL 式）把 Class-IL final 0.31→**0.858**、遺忘 0.50→**0.06**、retention>1.0（3 seeds, std 0.003）。診斷正確：病灶是線性頭的 recency/magnitude bias，換成無偏原型讀出即解。候選清單裡的 cosine/BiC/class-balanced replay 尚未試（NCM 已夠強，這些可作為進一步小幅優化或在更大規模時備用）。

### P2 — Conflicting-task benchmark（任務真正衝突）
- **為什麼**：目前所有任務骨子裡共享同一函數（sum→bucket），head/adapter 吸收差異，使「不遺忘」異常容易（retention→1.0）。真實 CL 的任務彼此衝突：學 B 會主動破壞 A 的共享層解。這是檢驗目前好結果是不是「benchmark 太友善」造成的關鍵。
- **做法**：在 `benchmark.py` 加一個 mode（或參數），讓每個 task 的底層 input→class 函數**真的不同**：例如 task 之間輪換不同聚合函數（sum / max / 特定位置的值 / parity）或不同 K，使共享層面臨衝突的特徵需求。仍用多頭 Task-IL（隔離 head，純測共享層衝突）。
- **驗收**：在新 stream 上跑 ReplayEWC vs DarkReplayEWC vs Joint。若 retention 從 ~1.0 明顯下降，就找到了「友善 benchmark」的邊界；看哪個機制最抗衝突。

### P3 — Buffer-free / generative replay
- **為什麼**：DER++ 仍存 2000 筆原始樣本；真正的終身學習可能不准存原始資料、串流無上限。
- **做法**：實作不存原始輸入的版本：
  - **feature replay**：只存/重播倒數第二層特徵（a2）而非原始 X；或
  - **小型 generative replay**：每類別一個 Gaussian（mean+cov）在輸入或特徵空間合成舊資料；
  - 對照原始 buffer 的 DER++/ReplayEWC。
- **驗收**：在 label_permuted 130-task 上，buffer-free 版本 retention 能不能接近原始 DER++ 的 ~1.0。

### P4 — Task-free（無邊界）CL
- **為什麼**：目前都靠 `on_task_end`（算 Fisher、更新 adapter/GPM 基、Class-IL 切片）——等於知道任務何時切換。真實串流沒有邊界。
- **做法**：拿掉 `on_task_end` 依賴：純 reservoir replay 本來就 boundary-agnostic；EWC 改成線上 Fisher（每步用 running estimate）或改用 replay-only。可選加上 online 漂移偵測。
- **驗收**：對照「給邊界 vs 不給邊界」的 ReplayEWC/DER++，量化失去邊界知識的代價。

## 3. 建議順序與理由（P1 已完成）

1. **P2（衝突任務）** — 現在最高價值：目前 Task-IL/Class-IL 都接近上界，但所有任務骨子裡共享同一函數，retention~1.0 可能是 benchmark 太友善。先驗證這點，才知道前面結論的可信邊界。
2. **P3（buffer-free）** 與 **P4（task-free）** — 把方法推向更接近真實終身學習的約束。

## 4. 慣例

- 每個實驗存成 `results_<描述>.json`；多 seed（≥3）。
- 新 trainer 加進 `trainers.py` 的 `TRAINER_REGISTRY`，`test_smoke` 會自動覆蓋。
- 跑完更新 `report.md`（新增小節 + 更新 §0 TL;DR 與結論）與本檔（移除已完成項）。
- 誠實第一：負結果（像 GPM、DER++ 在 Class-IL 反轉）和正結果一樣要記錄。
