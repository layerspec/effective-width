# Network Deconvolution（Ye et al. ICLR 2020）被引清單掃描（2026-10-13）

來源：Semantic Scholar Graph API，`paper/arXiv:1905.11926/citations`，10-13 07:50 CST 抓到 **42 筆**（OpenAlex 只列 13 筆，W2947556543；用 S2 的）。
目的：T3 筆記 §1 的「無人量白化後寬度、無人連到 ⌈τC⌉」這句要有被引清單背書；順便找比 ND 更新的 patch 白化運算子。
讀的深度：**只有 Cho 2021 讀了全文**（`refs/T3/cho2021_conv_unit_whitening_2108.10629.pdf`，文字抽出）；其餘只看 S2 的題目／場所，星號者另看了 arXiv 摘要。

## 1. 與 T3 有關的（7 筆）

| 文獻 | 做了什麼（讀到的深度） | 對 T3 |
|---|---|---|
| **Cho, Cho, Kim, Kim, ICCV 2021, Improving Generalization of Batch Whitening by Convolutional Unit Optimization**（arXiv 2108.10629；DOI 10.1109/ICCV48922.2021.00527，pp. 5301–5309，Crossref 核；全文） | 前提句與我們同：「whitening theory assumes the input of the linear layer is whitened」；把 IterNorm（**通道側**白化、C 維）從 conv 之後搬到 conv **之前**、去掉仿射 γβ；量 (i) conv 輸入的平均通道相關 ρ（式 4）、(ii) IterNorm 輸入的 **rank(X^l)/C^l**（式 5；rank 門檻未寫），ResNet-56 CIFAR-100：原單元 conv 輸出**不滿秩**、他們的單元（激活後再白化）**滿秩**（圖 3a）；主張白化穩定性來自輸入滿秩 | **最近的鄰居**。差別：(a) 通道側（C）不是 patch 側（C·k²），所以不是 T3.1 的最大化子；(b) 量的是秩/C（無 τ、無 r_max），不是 τ 寬度；(c) 沒有核正交、沒有 ⌈τC⌉；(d) 「conv 輸出不滿秩、ReLU 後滿秩」是我們 ReLU 步變寬的一個獨立旁證（通道秩版）。論文寫 T3 時引：「Cho et al. move channel whitening in front of the convolution on the same premise and report the channel rank of its input; the patch-level operator and the τ-width ceiling are not in their paper.」 |
| Huang, Zhou, Liu, Zhu, Shao, CVPR 2021, Group Whitening（arXiv 2009.13333*） | 通道側分組白化；分析正規化對特徵的約束與「representational capacity」（摘要未提秩／維度） | 通道側；capacity 是指約束自由度，不是 τ 寬度。只提一句 |
| Weng, Huang, Zhao, Anwer, Khan, Khan, NeurIPS 2022, An Investigation into Whitening Loss for SSL（arXiv 2210.03586*）；同作者 TPAMI 2024「Understanding Whitening Loss in SSL」 | SSL 嵌入的白化損失實際只要求嵌入**滿秩**就能避免維度塌縮 | 對象是嵌入向量，不是卷積 patch；「白化 ⇒ 滿秩」與 T3.0 的 ⌈τC⌉ 同方向，但他們沒有 τ 計數。文獻定位用 |
| Yong & Zhang, ECCV 2022, Embedded Feature Whitening（DOI 10.1007/978-3-031-20050-2_20；搜尋結果） | 白化放在**梯度**（ZCA 矩陣改權重梯度），架構不動 | 預條件器，不改表徵；與 T3 無關，不引 |
| Liu et al. NeurIPS 2021, Convolutional Normalization（arXiv 2103.00673；題目） | 卷積的 Fourier 域正規化 | 核側；若改稿要列正交卷積家族時再讀 |
| Su et al. ICML 2022, Scaling-up Diverse Orthogonal Convolutional Networks by a Paraunitary Framework（題目） | 正交卷積的統一框架 | 核側（ONI 的親戚）；同上 |
| Weng et al. ICLR 2024, Modulate Your Spectrum in SSL（arXiv 2305.16789；題目） | 嵌入譜的調制 | 嵌入側；不引 |

其餘 35 筆：應用（手語、作物病害、除霧、雷達、聲源定位 ACCDOA 系列、超解析、影像壓縮、缺值補值、材料、蛋白質）、綜述（Huang 2020 TPAMI 正規化綜述 489 引、Neural Trees 綜述）、重現研究 2410.01189（已在 T3 筆記）、Neural Parameter Allocation Search／Shapeshifter、random-labels（NeurIPS 2020；引 ND 談共變異數對齊）、事件相機光流、Controllable Feature Whitening（ICCV 2025，偏差消除）、一筆無效記錄。沒有一筆量白化後的 τ 寬度或把白化連到 ⌈τC⌉。

## 2. 結論（寫進 T3 筆記 §1 與文獻總帳）

- 「無人量白化後寬度」要改成：**無人量 patch 白化後的 τ 寬度**；通道秩/C 有 Cho 2021 量過（IterNorm 輸入、ResNet-56 CIFAR-100）。
- 「白化輸入」這個前提句在 Cho 2021 已明寫；我們的新點是 patch 側的運算子（ND）＋核正交＋⌈τC⌉ 的夾擠，寫時要說「on the same premise」。
- 更新的 patch 白化運算子：清單裡**沒有**；ND（2020）仍是 patch 側唯一的發表運算子。
- 未讀：Cho 2021 的 rank 門檻（numpy matrix_rank 預設？）與圖 3a 的數值；Group Whitening 的 capacity 分析段；Convolutional Normalization 與 Paraunitary 的正文（只在改稿列正交卷積家族時需要）。
