# 地圖 L5：寬度的選擇與壓縮（2026-09-30 第一輪；大部分已在 `related-work-boundary-2026-09-22.md`、`p2m-prior-art-2026-09-30.md`、`a30-prior-art-2026-09-30.md`）

**問題**：寬度怎麼被選、學、剪？學出來的寬度長什麼樣？無標籤的判準有沒有人做？

## 1. 已知

| 主題 | 文獻 | 陳述 | 與我們的關係 |
|---|---|---|---|
| 剪枝的價值是架構 | **Liu et al., ICLR 2019**（1810.05270） | 結構化剪枝得到的小網路從頭訓與微調相當；自動找出的寬度從頭訓也追得上；「剪枝的價值在於找到架構」 | 我們的 A18／P2（尺定寬重訓）就是這個範式；他們已說明「寬度是可以事後從頭訓的東西」 |
| 逐層寬度的判準（有標籤） | CHEX、DMCP、TAS、BCNet、Network Adjustment、AutoSlim、MetaPruning（`p2m-prior-art`） | 以準確率、KL、FLOPs 利用率、supernet 搜尋定每層寬度；ImageNet ResNet-18 1.0–1.2G 上比均勻高 2.5–2.8 點；**形狀：FLOPs 給後段** | 均勻是弱基準；成本項與訊號項要分開 |
| 逐層寬度的判準（無標籤、資料相依） | **Garg 2019**（PCA 99.9%，CIFAR）、**Garg 2024**（utilized rank，min(m,d) 分母）、**Yu & Wu 2023**（輸出 PCA、AFM 摺入）、Chee 2022（ID＋摺入）、Maile 2022 NORTH\*（ε-數值秩觸發生長） | 用激活的 PCA／秩定寬度並摺入或重訓，CIFAR 與 ImageNet 都有 | 我們的尺與摺入運算子屬於這一族；差別在 r_max、門檻、τ、隨機對照、跨配方 |
| 寬度的規律性 | Width Transfer（Chin 2021）：最佳化出的相對寬度在深度、寬度、解析度、資料量之間形狀相近 | | 與我們「剖面跨配方一致」同型 |
| 訓練中調寬度 | Yuan 2023（無訊號、等比例長）、GrowEfficient、PruneTrain、Shin 2025（動態秩）、NORTH\* | 訓練中長／剪的排程與省算力的數字 | 控制器（已降為一段）的相關工作 |
| 訓練後的低秩可壓縮性 | Garg 2024（ViT-B MLU 35%）、Yu & Wu（DeiT-B −20% −0.09）、Denton 2014、Jaderberg 2014、Idelbayev 2020 | 訓練後的層可低秩分解而不掉分 | 「閒置寬度」在文獻裡叫低利用率／可壓縮性，多類任務也有（判準較鬆） |

## 2. 留下
- **無標籤、固定 τ 的尺與有標籤判準之間的差**沒有系統比較（Garg 2024 與我們讀數的差是這一格）。
- **成本項對訊號項**沒有人拆開報（`p2m-prior-art` §4）。
- 學出來的寬度為什麼是「後段寬」：文獻只有實證（每通道 FLOPs 便宜）；與 L2 的「後段塌縮」看似相反——**壓縮文獻說後段要留寬（參數多、FLOPs 便宜），
  秩文獻說後段秩低**。兩者不矛盾（低秩但通道多），但論文要講清楚寬度 ≠ 秩。

## 3. 給我們的
- (ii) 摺入與 (iv) 閒置寬度的措辭約束已記在 `a30-prior-art` §4 與 `pending-text-corrections`。
- 「寬度 ≠ 秩」這一句是 L5 給脊椎的：可調撥的寬度由秩決定（我們），該留的寬度由成本決定（他們）。

## 4. 09-30 引用圖掃描補入（Ansuini 的 457 篇被引；讀原文）

**Chen, Bu & Yan, NeurIPS 2024,「Unveiling the Matthew Effect Across Channels: Assessing Layer Width Sufficiency via Weight Norm Variance」**（37: 60040–60060；PDF `refs/scan0930/matthew_neurips2024.pdf`）
- **陳述**：每層各通道的權重範數 ‖w_i^l‖·‖w_i^{l+1}‖ 的**跨通道變異數**在訓練中的走勢分兩型：IS（升到飽和＝層夠寬、有很多相似通道）與 DS（升後降＝層太窄、通道近正交）；
  三階段（初期 cos(∇, W) ≈ 0 變異不動 → 中期急升 → 後期分型）。VGG-16／ResNet-18 CIFAR-10：**前幾層與後幾層 IS、中段 DS** → 提出「流線型」寬度：中段加、後段砍
  （VGG-16 [64,96,128,160,256,256,320,512,384,384,320,256,256]，參數 61.5%、FLOPs 104%，CIFAR-10 91.02→93.41、CIFAR-100 66.17→68.63、TinyImageNet 49.8→54.4，10 種子）。
- **與我們的關係**：(a) **同一個結論、不同的尺**：他們用權重範數變異數說 VGG-16 後段太寬，我們用共變異數 τ 寬度讀出後段閒置（P1 摺入 67% 參數不掉分）——
  兩把互不相干的尺在同一網路指同一處，是收斂證據，也是要引並劃界的前作（L5「無標籤判準」列要加一行）。(b) 他們的尺**只能訓練到收斂後才判**（自己在結論寫的限制），
  我們的尺在訓練中就讀（控制器段的差異化）。(c) 他們自己把中段 DS 連到 Ansuini 的 ID 駝峰（「中段 ID 高 → 通道正交」），是 L4 那條「ID 對寬度」的猜想，無理論。
  (d) 他們的 baseline 弱（VGG-16 CIFAR-10 91.02 對我們 93.83、lr 每 40 epoch ×0.1、wd 1e-4），「+2.4」不能直接對我們的數字。(e) **§4.1 的 cos ≤ 0.0003** 是 B6 的前作觀察（見 plan §9.25 補記）。
  (f) ResNet-50 他們把後 8 個 block 的 3×3 從 512 砍到 192–256 而不掉分（CIFAR-100）——與 A26／P2i「殘差網路在 τ=0.999 讀滿」不矛盾（他們的是 CIFAR 上的 ResNet-50，通道 2048；我們是 ResNet-20 通道 64）但要在文中說清楚「讀滿 ≠ 不能砍」。
- **給論文**：Related Work 的 L5 段加一句；`sec:a18`（摺入）對照一行；B6 段引 0.0003；控制器段的「訓練中可讀」差異化。

其他掃到、下載未讀（`refs/scan0930/`）：Mudarisov–Burtsev–State 2026（2608.02064，transformer FFN 逐層寬度配置，幾何量）；Dees–Agarwala–Lowman 2022（2201.11813，autoencoder 特徵值訓練前後）；
「Structured Initialization for Vision Transformers」NeurIPS 2025（2505.19985，A33 的相關）；「The Residual Stream's Effective Depth」2026（2609.31098，L6）；
「Causal Dimensionality of Transformer Representations … Layer Structure」2026（2605.08740）；「Flatness is Necessary, Neural Collapse is Not」NeurIPS 2025（2509.17738）；
Bridging mean field and finite width with BN（2205.13076；**L1 的第 1′ 環，先讀**）。
