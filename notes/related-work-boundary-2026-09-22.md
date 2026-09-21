# 劃界：零重訓摺入與訓練中調撥的直接先例（2026-09-22，審稿 09-20 T6／S1／S8／S9／S10；稿子不動，先把數字放在一處）

來源都是原文 PDF（arXiv 2203.15794 CHEX、2108.00065 Chee、1707.06168 He、2306.12700 Yuan）用 pdftotext 抓的表；我們的 FLOPs
用 `scripts/p2_flops.py` 的 `macs()` 從 `results/cssp/fold_results.csv` 的寬度算（全寬 313M MACs）。

## 1. 零重訓摺入（P1）—— 我們的運算子是既有做法

| 方法 | 選通道 | 重建 | 微調 | 各層砍多少 | VGG-16 CIFAR-10（不微調） | ImageNet |
|---|---|---|---|---|---|---|
| He, Zhang, Sun ICCV 2017 | LASSO（以下一層輸出重建誤差為目標） | 最小平方重建下一層 | 有（10 epoch）；也報不微調 | 手訂（淺層砍重） | ResNet-56 2×：不微調 +2.0、微調 +1.0（基線 92.8） | VGG-16 top-5 誤差 2×／4×／5×：不微調 +2.7／+7.9／+22.0；微調 0／+1.0／+1.7 |
| ThiNet（Luo 2017） | 貪婪，以下一層重建誤差選 | 最小平方 | 有 | 手訂 | — | VGG-16 −46% FLOPs、66.59 top-1、30 epoch（Chee 表 2 轉引） |
| **Chee, Renz, Damle, De Sa NeurIPS 2022** | interpolative decomposition（座標子集＝**CSSP**） | 插值矩陣摺入下一層 | **無**（也報微調） | 由單層泛化界的啟發式自動定 | **基線 93.6；砍到 35% FLOPs 不掉分**；50% FLOPs ID+LRank 93.43（Δ 0.31）0 epoch；ID 微調 60 epoch 93.31 | VGG-16 −43% FLOPs、68.84（Δ 2.67）0 epoch；−25% FLOPs 68.0 |
| **我們 P1**（`results/cssp/`） | 貪婪 CSSP（pivoted Cholesky）於 ReLU 後共變異數 | 線性重建摺入下一層 | **無** | 尺 k\*(τ) | 基線 93.83；**τ=0.999：−12% FLOPs（67% 參數）93.82；k\*(0.99)：−31% FLOPs（39% 參數）93.34（Δ 0.49）；τ=0.95：−58% FLOPs 88.87（Δ 4.96）** | ResNet τ=0.999 砍 1–3% 參數 Δ ≤ 0.4；ε=0.01 砍 4–8% Δ 0.2–1.2；ViT-B 92.5% 參數 −0.28 |

**誠實讀法。** 同網路同資料集、同樣不微調：Chee 砍 35% FLOPs 不掉分，我們砍 12% 不掉分、砍 31% 掉 0.5。**我們的運算子不比 Chee 好，
而且就是同一個運算子**（座標子集＋插值摺入；他們的「ID」= 我們的「CSSP＋重建」；S8 屬實）。他們連「各層砍多少」也自動定了。
我們與他們的差只有三點，論文只能寫這三點：
1. 砍的寬度不是搜尋出來的，是**尺讀出的**（固定 τ，訓練前就能算的統計量的同一族）——但 Chee 的啟發式也不搜尋。
2. **跨架構表**（VGG／ResNet-18/50／ViT-B／ConvNeXt-T／DeiT-S 同一個 τ）——Chee 只有 VGG-16、MobileNetV1、3D-Conv。
3. 命題 A 的比值（1.02–2.84）——這是 Deshpande 2006／Çivril 2009 的界在此的實測，不是新界。
S9 的「子空間 92.92 vs CSSP 88.90」正是 Chee 的 ID+LRank（座標子集＋低秩）觀察；要引。

**論文措辭（待稿子解凍）。** 「The shrink operator is the interpolative-decomposition fold of Chee et al. [NeurIPS 2022]
(column subset + interpolation matrix into the next layer; He et al. [ICCV 2017] and ThiNet with LASSO / greedy selection and
least-squares reconstruction are its fine-tuned ancestors). We add nothing to the operator; we use it at the width the ruler
names, without fine-tuning, on seven architectures, and read its residual against Proposition A.」P1 段的「零重訓摺入」不得寫成貢獻；
貢獻是「尺的寬度可以直接用」。

## 2. 訓練中調撥（P2）—— 直接先例與同設定數字

| 方法 | 訊號 | 動作 | 跨層重分配 | 訓練前起點 | 一次訓練 | 同設定數字 |
|---|---|---|---|---|---|---|
| **CHEX**（Hou et al. CVPR 2022） | 權重矩陣的 column subset selection（層內）＋再生階段的重要度 | 反覆剪與再生 | **有**（全域稀疏約束下重分配各層通道數） | 無（任意起點） | 是 | ImageNet ResNet-50 25% FLOPs 76.0；ResNet-18 1.04G（−43%）；CIFAR-10 **WRN-28-10** −74.8% FLOPs 96.2%（＝基線）、訓練成本省 48.8%；**無 ResNet-20／VGG-16 CIFAR** |
| GrowEfficient（Yuan et al. NeurIPS 2021） | 結構化連續稀疏化（LASSO 類） | 長 | 有 | 無 | 是 | WRN-28-10 CIFAR-10 −71.8% FLOPs 95.3%、省 67.9%；ImageNet ResNet-50 −49.5% 75.2 |
| PruneTrain（Lym 2019） | 群 LASSO | 剪 | 有 | 無 | 是 | ImageNet ResNet-50 −44% 75.0、省 30% |
| **Yuan et al. NeurIPS 2023** | **無訊號**：固定 9 階段排程、各層等比例 | 只長 | 否 | 1/4 寬 | 是 | **CIFAR-10 ResNet-20 160 epoch：54.9% 訓練成本 92.53±0.11 vs 92.62±0.15；最終全寬（推論 100%）**；VGG-11 52.9% 92.34 vs 92.14；ImageNet ResNet-50 60.1% 75.90 vs 76.72 |
| Early-Bird（You et al. ICLR 2020） | BN γ 的通道遮罩在 10–20% 訓練後穩定 | 一次剪後訓 | 否 | 無 | 是 | 觀察與我們「epoch 15 動一次」同款；要引 |
| NORTH\*（Maile et al. AutoML 2022） | 後激活 ε-數值秩 ÷ 寬度 | 只長 | 否 | 無 | 是 | VGG-11 CIFAR-10 少於一半參數贏靜態 |
| SRigL（ICLR 2024） | 稀疏訓練 | 固定 FLOPs 下結構化剪／長 | 有 | 無 | 是 | 要引 |
| **我們 P2／P2g／P2i** | **無標籤的激活共變異數＋未滿足變異（patch 共變異）** | 縮（CSSP 摺入）＋放（未滿足變異方向、函數保持） | 有（注水） | **有**（A′：epoch 0 動作） | 是 | VGG-16 CIFAR-10 參數預算 0.673：94.11（1.22× 算力）；FLOPs 守恆 P2g 待收；**ResNet-20 P2i 待收（f47：訓練成本 0.55、推論 0.47，對 Yuan 的 54.9%／100%）** |

**誠實讀法。** 與 CHEX 的差別只剩四項：訊號無標籤（前向統計，不用權重或梯度）、放的方向由未滿足變異定、訓練前起點、三個命題；
「CSSP 縮＋放＋跨層重分配＋一次訓練」四件事 CHEX 都有。同設定數字：CHEX 沒有 CIFAR ResNet-20／VGG-16，做不了同設定表；
能同設定的只有 **Yuan 2023 的 ResNet-20**（P2i）與 Chee 的 VGG-16 CIFAR-10 微調後表（50% FLOPs：ID 93.31／HRank 93.43／Polar 93.92，
各多 60／480／200 epoch；我們 P2g f60 是一次 100 epoch，0.60 MACs——**這是可以放同一張表的列**）。

**論文措辭（待稿子解凍）。** Related Work 加一段：「CHEX [CVPR 2022] already interleaves column-subset pruning with regrowth and
redistributes channels across layers in one training run. Our controller differs in what it reads (label-free second-order
statistics of the forward pass, not weights or gradients), in where it grows (the unmet-variance directions of Proposition B),
and in that it can act before training (§…); it does not differ in the mechanics.」以及一句 Early-Bird。**不得寫「first」**。

## 3. 對審稿筆記的回填

- T6：已劃界，數字如上；狀態從「重大」改「已引用、貢獻縮為尺寬度＋跨架構」。
- S1／S8／S10：屬實，措辭如上；S9：ID+LRank 就是子空間版，要引。
- 同設定表的空格：P2g f60（對 Chee 表 1）、P2i f47（對 Yuan 表 2）——兩條隊列跑完填。
