# 當前研究方向備忘：P13 全域可分表徵

這份檔案只保留目前可行的下一步。完整研究日誌以 `report.md` 為準，待辦與命令以
`NEXT_STEPS.md` 為準。

---

## 1. 目前狀態

主線已從 pi digits 的抗遺忘，推進到 Split-CIFAR-100 / ResNet18 / 無 task-id
部署。P11/P12 的關鍵結論是：

- task-IL 下，會動 ResNet18 + replay/DER++ 能累積表徵，DER++ retention 約 0.64。
- 拔掉 task id 後，class-IL 大幅掉落：DER++ 線性頭約 0.17，NCM 約 0.235。
- BiC 只修一部分 task-group bias，cosine head 有害，NCM 仍是最好 post-hoc 讀出。
- 因此缺口不是「讀出器不夠好」，而是會動 backbone 沒學出全域 100-way 可分表徵。

方向判斷：**下一步應改訓練目標，不應再調 post-hoc 讀出或 DER++ gate。**

---

## 2. 已停損或低優先方向

- P2.5-P2.9 的 adaptive DER++ gating 已有足夠負面結果；局部/短視訊號無法可靠判斷
  DER++ 長期收益，外部效度低。
- P12 的 cosine/BiC/NCM 已測出 post-hoc 讀出的上限；繼續做讀出微調不是最高槓桿。
- P9/P9b 已釐清 buffer-free 累積：PNN 與 softmax-KD 可行，但仍低於 replay/DER++；
  這是獨立支線，不是目前 class-IL 部署缺口的直接解。

---

## 3. P13 實驗順序

目標：把 DER++ class-IL NCM 從約 0.235 往 task-IL 0.641 推近，同時保留 task-IL
forward-transfer / retention 指標。

建議順序：

1. **P13a：supervised contrastive loss（✅ 已做，負面結果，見 report §27）**
   - 已掛 `--supcon-weight`/`--supcon-temp`，對 [current ∪ replay] penultimate features 算 SupCon。
   - DER++、resnet18、class-IL：`supcon=0.03` 幾乎等於 baseline（seed0 NCM
     0.236 vs 0.239），`0.1` 已下降（seed0 NCM 0.217），`0.5/1.0` 兩 seed 明顯更差
     （NCM 0.237→0.209→0.194、forgetting 0.455→0.469/0.480）。
   - **病灶＝buffer 每類稀疏**：replay 32 散在多達 100 類、舊類湊不出同類正對，SupCon 只
     收緊當前任務簇、又與 CE/DER++ 搶容量。否證的是「在稀疏 replay batch 上直接做」，
     不是 SupCon 概念本身 → 改走 P13b 補足 per-old-class 正對。
   - 結果檔 `results_p13_derpp_supcon{0,0.5,1.0}_resnet18.json`、
     `results_p13a_derpp_supcon{003,01}_resnet18_seed0.json`。

2. **P13b：prototype memory bank / proto-contrastive（✅ 已做，診斷正確但仍負面，見 report §28）**
   - 實作 `ProtoBank`（持久 per-class 原型，current+replay EMA 更新）+ `--proto-weight`。
   - DER++、class-IL 掃 proto ∈ {0,0.1,0.5,1.0}：NCM 0.237→0.233→0.221→0.207。
   - 對照 P13 SupCon 同劑量 NCM 0.237→0.209→0.194：**原型庫把傷害變小（證明 P13「正對稀疏」
     診斷正確），但仍無任何劑量淨增益**。修好正對稀疏是必要不充分；auxiliary 對比目標抬不動
     class-IL，瓶頸是表徵起點/容量 → 直接做 P13c。
   - 結果檔 `results_p13b_derpp_proto{0,0.1,0.5,1.0}_resnet18.json`。

3. **P13c：pretrained init + fine-tune（⬆ 升為最高優先）**
   - P12/P13/P13b 三方向（讀出、batch 對比、原型對比）全撞 ~0.235 上限；§18 frozen ImageNet
     NCM 0.530 vs 從零 0.237 的 2× 差距最強指向表徵起點/容量。
   - 問題是從廣泛預訓起步後，continual fine-tune 能否兼具 task-IL 累積與 class-IL
     全域可分性。

---

## 4. 當前工程注意

- `run_backbone_transfer.py --eval-mode classil` 已有線性 / NCM / cosine / BiC 四種讀出。
- P13 應只改訓練端，結果格式盡量維持現狀，方便和 `results_p12_*_resnet18.json` 比。
- 本機 `.venv/bin/python` 已可用 CUDA（torch 2.12.1+cu130、RTX 2070 sm_75）；ResNet18
  20-task × 2-seed class-IL 一輪約十餘分鐘。
