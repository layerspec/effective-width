# 動態寬度文獻清場（2026-09-16，第一輪，半天）

目的：路 1＋路 2 合併題目「寬度不該是超參數」要贏的對手是誰、他們的訊號是什麼、曲線在哪裡、我們哪裡不同。
**結論先寫：題目沒有被做掉，但兩個最近的對手（AWN、Yuan 2023）把「口號」與「訓練省算力」各佔了一半；
我們能站的位置是「卷積層、逐層、訓練前有起點、訊號是無標籤二階統計、ImageNet 上同準確率省訓練算力」。**

## 對手表

| 方法 | 出處 | 訊號（何時／何處） | 長／縮 | 逐層？ | ImageNet | 省的是什麼 | 與我們的差 |
|---|---|---|---|---|---|---|---|
| **AWN** Adaptive Width NN | Errica et al., **ICLR 2026**, arXiv 2501.15889 | 每個神經元一個可學的重要度分布（變分推論），寬度 = 分位數；靠 backprop | 兩者 | 是 | **無**；ResNet-20 只調分類頭 MLP，conv 不動 | 省「調寬度的超參數搜尋」；單次訓練反而慢 1.2–3.9× | 口號「寬度隨任務難度自適應、減少超參數」跟我們重疊。他們沒碰卷積、沒有 r_max、沒有訓練前起點、不省訓練算力。**論文要正面引用並劃界** |
| **Yuan et al.** Incrementally growing, variance transfer + LR adaptation | NeurIPS 2023, arXiv 2306.12700 | **無訊號**：固定排程（9 階段、指數式 epoch 分配），各層等比例放寬 | 只長 | 否（全網同比例） | **ResNet-50 訓練 FLOPs 60.1% → 75.90 vs 76.72**；MobileNetV1 63.7% → 69.91 vs 70.80 | 訓練算力 40% | 這是要打的曲線。他們的弱點：何時長、哪層長都是人訂的；我們的訊號決定何處 |
| GradMax | Evci et al., ICLR 2022 | 新神經元初始化使梯度範數最大（SVD）；何時長是排程 | 只長 | 初始化逐層 | MobileNetV1 68.6 vs random 66.9（訓練 86.7% 成本） | 訓練算力 | 初始化技術，可以直接借用（函數保持） |
| Firefly | Wu et al., NeurIPS 2020 | 泛函鄰域內最陡下降，每 N epoch 長；Taylor 近似選候選 | 長（寬與深） | 是 | ResNet-50 71.2% 成本 → 75.01 | 訓練算力 | 局部最佳化成本高；Yuan 已勝 |
| Net2Net | Chen 2016 | 無訊號，分裂複製 | 長 | — | 60.1% → 74.89 | — | 基線 |
| NORTH\* | Maile et al., AutoML 2022 | **後激活矩陣的 ε-數值秩 ÷ 寬度**（我們的統計量家族）觸發；正交初始化 | 只長 | 是 | 無；VGG-11 CIFAR-10 少於一半參數贏靜態 | 參數 | 最近的親戚。差：無 r_max 正規化、無訓練前起點、無縮、無 ImageNet、無理論 |
| Self-Expanding NN／SECNN | Mitchell et al. 2023；Aleksandrov 2024 | Natural expansion score（自然梯度下損失下降量）；有上界證明 | 長（寬與深）；MLP 版可縮 | 是 | 無；CIFAR-10 84.1% | 參數 | 訊號要二階梯度資訊；我們的是前向統計 |
| Growing Tiny Networks（TINY） | Verbockhaven et al., TMLR 2024；DAG 版 ESANN 2025 | backprop 的「表達力瓶頸」（泛函梯度無法被架構表達的部分） | 長 | 是 | 無 | 參數 | 理論漂亮（泛函梯度），規模小 |
| **DemP** Maxwell's demon | Dufort-Labbé et al., TMLR 2025, arXiv 2403.07688 | **神經元飽和／死亡**（幾次前向即可讀出）＋噪音注入與正則化促進死亡 | 只縮（dense→sparse 結構化） | 是 | 有；訓練加速最高 3.56× | 訓練算力＋推論 | 縮的方向與我們同源（飽和），但他們製造死亡再剪；我們讀秩 |
| MorphNet | Gordon et al., CVPR 2018 | BN γ 的 L1 收、均勻放，FLOPs／size 正則 | 兩者（交替） | 是 | 有 | 推論 FLOPs | 訊號是權重大小 |
| Slimmable／AutoSlim／OFA | Yu 2019；Cai 2020 | 一次訓練多寬度（supernet） | 推論時選 | 是 | 有 | 推論 | 不同問題（部署多檔），但審稿人會要比 |
| SWE＋Gradient Voting | arXiv 2509.18842 (2025) | 新神經元與舊的共享權重避免死亡 | 長 | 是 | 待查 | 訓練 | 初始化技術 |

另：Nimble NN（Royal Soc. A 2025）、「Data classification with dynamically growing and shrinking NN」(2507.01043) 規模小，不構成對手。

## 要打的曲線（定量門檻）

1. **ImageNet ResNet-50**：Yuan 2023 在 60% 訓練 FLOPs 下 −0.8 點。我們要在 ≤ 60% 下 ≤ −0.3，或同 −0.8 下 ≤ 50%；
   而且說得出「哪一層在什麼時候長／縮、為什麼」。
2. **CIFAR-10 ResNet-20／VGG**：Yuan 55% 成本 −0.1 點；我們已有 VGG-16 ReLU 後尺 67% 參數持平（A18），但那是兩次訓練；
   一次訓練要追平。
3. **多類任務不縮、少類任務縮**：AWN 的「寬度隨難度」只有 MLP 頭；我們在 conv 全網、有 A24／A25 的類別數定律。
4. 推論算力另計（DemP、MorphNet 的場子）。

## 我們的位置（論文要這樣寫，不寫「他們沒做」）

1. **起點是算出來的**：所有成長法從任意種子開始、AWN 從任意寬度開始；我們從架構 × 影像二階統計算出每層初始寬度（三段定律）。
2. **訊號是無標籤的前向二階統計**：k\*/r_max 與對齊飽和度，從 Σ_out = WΣ_patch W^T 推出；不是梯度範數、不是損失下降量、不是 γ。
   NORTH\* 用同家族統計量但沒有 r_max、沒有理論。
3. **逐層、雙向、函數保持**：縮 = PCA 截尾摺進下一層；長 = GradMax 式（借用）。總參數或總 FLOPs 守恆。
4. **調撥時刻由對齊到位時間定**（epoch 7–15），不是超參數排程（Yuan 的弱點）。

## 待查（第二輪）
- DemP 的 ImageNet 表（網路、稀疏度、準確率）；SWE 的規模；Chase／結構化 RigL 的訓練加速數字。
- AWN 的 truncation 曲線與 CIFAR-100 那次「unlucky run」。
- 「訓練前決定寬度」的理論：搜尋只找到 Ω(n^rank(X)) 型的表達力界與 ID 估計，沒有人做「逐層初始剖面預測」；
  但要再查 mean-field CNN（Xiao 2018）與 conjugate-kernel 譜隨深度的文獻（見 `apriori-plan`）。
