# 文獻地圖第二輪，第二批（2026-10-01；PDF 在 `refs/scan1001/`、`refs/L4b/`，不進 git）

三次 WebSearch（逐層 collapse 剖面、殘差加法處的譜、訓練前的寬度剖面）＋ L4 待讀三篇。第一批（B7 的先行工作）在 `b7-prior-art-2026-10-01.md`。
兩欄：已經有人做的／給我們的。

## A. 與 B7（κ 對梯度 Gram）直接相關

| 文獻 | 已經有人做的 | 給我們的 |
|---|---|---|
| **Liu, Paquette, Sous 2026**（2605.05683）「Spectral Lens: activation and gradient spectra as diagnostics of LLM optimization」 | decoder-only LM（modded-NanoGPT，到 3.57B）：**trace 正規化的激活共變異數譜＋逐樣本梯度 SVD 譜**當診斷；同損失不同 batch size 的內部幾何不同；早期激活尾巴預測 token 效率；模加法玩具模型把譜的變化接到特徵學習與預條件 | **最接近 B7 量測的既有工作**（同時量 Σ_out 與梯度譜），但他們是逐樣本梯度（參數空間）、LLM、診斷用途，沒有「兩者之間的關係」（記憶窗、列空間）也沒有 τ 寬度對初始化的比較。論文 B7 段要引並劃界：「they read the two spectra side by side as diagnostics; we relate them」 |
| **Zhao, Cheng, Masarczyk, Lucchi, ICLR 2026**（2602.16642）「Optimizer choice matters for the emergence of neural collapse」 | 3,900 跑：**AdamW 不到 NC，SGD／Adam 會**；證明 SGD、SignGD＋coupled wd、SignGD＋decoupled wd 的 NC0 動力學不同；momentum 加速 NC | (1) 我們的 CIFAR 配方（SGD＋momentum＋wd）是最會塌縮的那種，A24／A25／A35 的「塌縮」要寫明優化器；(2) **公開 ImageNet 權重的配方差（torchvision SGD vs timm 的 AdamW／LAMB）也差在這裡**，但跨配方 T–T 仍 0.86——順序對優化器也穩健，這是一句可以寫的話；(3) B6 的「無駐點」是 SGD＋coupled wd 的陳述，decoupled wd 要另寫 |

## B. L2 訓練中的低秩／collapse（補）

| 文獻 | 已經有人做的 | 給我們的 |
|---|---|---|
| **Jacot, Súkeník, Wang, Mondelli 2024**（2410.04887；ICLR 2025）「Wide neural networks trained with weight decay provably exhibit neural collapse」 | 離開 UFM：末端 ≥ 2 個線性層的網路，在低訓練誤差＋平衡條件下證 NC；寬網路＋wd＋金字塔拓樸、光滑激活下條件成立 | 第 3 環「塌縮有定理」的最新一條；條件是**末端線性層**——我們的 conv 輸出在 ReLU 與池化之前，不在定理的對象裡，正好解釋「池化特徵塌（wcv 0.07）、conv 輸出不塌」（A35） |
| **Wang, Gai, Zhang 2024**（2405.00985）「Progressive feedforward collapse of ResNet training」 | ResNet **中間層**的 NC 程度隨深度單調（PFC），wd 下中間層特徵沿 Wasserstein 測地線（直線）到 ETF；多層 UFM | 中間層 collapse 在 ResNet 上的實證＋模型；量的是攤平後的類別幾何，不是位置解析的譜；與 Rangamani 2023 並列引在第 3 環 |
| **Rupa 2026**（2604.00230）「Neural collapse dynamics: depth, activation, regularisation, and feature norm thresholds」 | NC 的**起點**由倒數第二層特徵範數到達一個 (模型, 資料集) 專屬門檻決定；深度／寬度／wd 只改到達速率 | TPT 時序的另一個讀法；B7 說權重記得的是塌縮前的梯度，塌縮的時刻就是這個門檻——可以引來定義「塌縮前」 |
| Garrod–Keating–Thrampoulidis 2026；Beaglehole–Súkeník–Mondelli–Belkin 2024 | 見 `b7-prior-art-2026-10-01.md` §2 | 同上 |

## C. L1／L5 訓練前可算的量

| 文獻 | 已經有人做的 | 給我們的 |
|---|---|---|
| **Zhu, Zhao, Lu 2026**（2609.23087）「Neural Spectral Capacity: measuring and designing architectures from network specification alone」 | 每個權重矩陣的奇異值譜 → 線性高斯通道的互資訊 → 逐層相加成一個標量 NSC；**標準隨機初始化下由 MP 律閉式算出、不必實例化模型**；逐層可加 ⇒ 預算下的架構搜尋是背包問題（NSC-DP）；transformer | 「從規格算一個譜的量」已有人做，但對象是**權重**（i.i.d. 高斯、MP）、全網一個數、用途是架構搜尋。我們的 S2 是**激活**共變異數的逐層剖面、給定 Σ_patch、與訓練後比較——要在 (i) 段劃界，並說明為什麼激活譜不能用 MP（規則 1）。題目陳述要加這篇當「最近的鄰居」 |

## D. L4 量法（待讀三篇）

| 文獻 | 已經有人做的 | 給我們的 |
|---|---|---|
| **Gao, Trautmann, Yu, Santhanam, Ryu, Shenoy, Ganguli 2017**（bioRxiv 214262）「A theory of multineuronal dimensionality, dynamics and measurement」 | PR 當維度；定義 **neural task complexity (NTC)**，證明它是神經狀態空間維度的上界；隨機投影理論給「要記錄多少神經元」（∝ 任務參數數 × log NTC） | 「維度由任務的複雜度上界住、多記錄神經元不會變高」——與 A24（類別數）同方向的一個神經科學先例，可在 Discussion 引一句；PR 的理論出處 |
| **Del Giudice 2021**（Multivariate Behavioral Research 56(3)）「Effective dimensionality: a tutorial」 | ED 的定義（等效正交維數）、與 intrinsic dimensionality 的區別、估計量 n1（熵）／n2（PR）／n∞ 的比較與偏差 | Table S1 的量法分類可引它當教程出處；規則 4（四種維度分開）有現成的引用 |
| **Stringer, Pachitariu, Steinmetz, Carandini, Harris 2019**（Nature 571; bioRxiv 374090） | V1 對自然影像的反應譜是 n^−α、α ≈ 1；證明平滑的碼要求 α > 1＋2/d（d ＝ 刺激集的維度）；低維刺激集的指數更大 | 激活譜為冪律的理論理由（平滑性）；我們的譜也是冪律（規則 1），指數隨層變——可引來說「冪律是預期的，形狀才是量的對象」。**注意**：他們的 α 是特徵值對序號的指數，與我們 `powerlaw_alpha` 欄的定義要核一次再比 |

## E. L6 殘差（補）

| 文獻 | 已經有人做的 | 給我們的 |
|---|---|---|
| **Fernando & Guitchounts 2026**（2605.14258）「Dynamics of the transformer residual stream: coupling spectral geometry to network topology」 | 三個 LLM 的逐層 **Jacobian 完整特徵分解**：訓練裝進單調的非正規性梯度（Henrici 0.04 → 0.70）；累積 Jacobian 的有效秩；初始化時沒有 | 殘差流的「譜幾何」在 LLM 上有人做了，對象是 Jacobian 不是共變異數；L6 的句子「加法處的共變異數譜沒有人量過」仍成立，但要把這篇與 Gahtan 2026 一起列為鄰居 |
| Gadhikar & Burkholz 2022（RISOTTO） | 殘差 block 的動態等距初始化 | 已在舊筆記 |

## F. 對三軸

- 第 1 軸：L2 的 2024–26 新作補齊（Jacot 2024、Zhao 2026、Rupa 2026、Wang 2024、Garrod 2026、Beaglehole 2024），L4 待讀清單讀完，L1／L5／L6 各加一個最近鄰（NSC、Spectral Lens、Fernando）。
  **仍 3.5**：劃界句未寫進稿子、bib 未核、L5 的 2026 寬度選擇新作沒有新命中（三次查詢）。
- 第 2 軸：題目的兩個最近鄰出現——NSC（規格 → 權重譜標量）與 Spectral Lens（激活＋梯度譜當診斷）；兩者都不是「逐層激活剖面由什麼決定」。題目陳述要加這兩篇。
- 第 3 軸：不變；Zhao 2026 給了 B6 的優化器邊界（coupled vs decoupled）。
