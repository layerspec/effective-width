# 逐層維度剖面文獻掃描（2026-09-08）

對應 CLAUDE.md「下一步 3」（重寫 §I／§II 框架）之前的相關工作盤點。問題：
2014–2026 年間，哪些工作報告過卷積網路（或方法可移植的 transformer）**表徵**
的逐層維度／秩／有效秩剖面？它們各自讀哪個張量、用什麼估計量、有無未訓練
對照與訓練動態、報告什麼形狀？哪些我們還沒引、哪些跟我們的承重主張撞到？

**結論（先講）：**

1. 逐層剖面的文獻分成四個互不引用的圈子：(i) 流形內在維度（Ansuini／Recanatesi／
   Cohen／Konz）、(ii) 線性秩與神經崩塌（Rangamani／Masarczyk／Harun／Feng）、
   (iii) SSL 表徵秩指標（RankMe／α-ReQ／LiDAR，**單層**，不是剖面）、(iv) 神經科學
   模型的有效維度（Elmoznino／Chen & Bonner，**全域池化後**）。沒有一篇把分母
   設成可達秩；把秩除以標稱寬度的只有 transformer 圈（Everett 2026、Jha 2025）
   與 RankMe 的 2048 上限。**r_max 作為分母仍無先例**，與 `rmax-novelty` 結論一致。
2. **兩個直接先行、目前零引用、必補**：Cohen et al. 2020（Nat. Commun.）明寫
   「pooling operations decrease both manifolds' radii and dimensions」，是我們
   「池化壓低水準」的流形版先例；Masarczyk et al. 2023（NeurIPS，tunnel effect）
   在 VGG-19／CIFAR-10 上量到數值秩在**前 75 步**就塌到類別數並維持到訓練結束，
   直接先於我們「剖面在 epoch 40 定型」的觀察。兩者都不矛盾，但不引就是硬傷。
3. 唯一算得上「撞到」的是 Schulte & Rügamer 2026：證明 Lipschitz 網路的內在維度
   **逐層不可能上升**，所以 Ansuini 式的駝峰是估計量假象。這條打的是 TwoNN，
   不打線性秩（非線性可以把線性秩撐高，ReXNet 已講），但我們寫規則 4 時要把這條
   放進去，否則審稿人會拿它來質疑任何「上升」措辭。
4. 訓練動態這一列要小心：Huh 2023 看到有效秩先收縮再回升、Stephenson 2021 看到
   深層流形維度在**晚期** epoch 因記憶化而變、Ansuini 末層 ID 非單調。我們的
   「epoch 40 定型」必須連 LR 排程一起寫，並限定於 plain VGG／CIFAR。

---

## 1. 範圍與方法

- 範圍：報告**表徵**（激活／特徵）逐層維度、秩或有效秩的工作；SSL 秩指標雖是
  單層也納入（任務要求）；剪枝文獻只取量激活秩的（HRank 及後繼）。
- 排除但列於附錄 A：量 Jacobian 或權重秩、量可分性（NCC／separation fuzziness）、
  量密度峰（Doimo 2020），以及純 transformer 且方法不可移植者。
- 方法：WebSearch（僅美國索引）＋ arXiv PDF 全文（pdftotext）＋ Crossref API 查
  DOI。全文能開的都開了；只讀摘要的在 §6 列出。引文以原文英文為準。
- 已引（`paper/refs.bib`）：ansuini2019、recanatesi2019、rangamani2023、papyan2020、
  feng2022、elmoznino2024、garg2019、lin2020hrank、daneshmand2020、kong2022、
  ghosh2022、nanda2023、han2021rexnet、kim2018rank1。以下表格仍列它們，但
  §3 只列**未引**的。

---

## 2. 比較表

欄位：讀取張量（conv 輸出／post-activation／block 輸出／池化／攤平）；估計量
（樣本＝位置或影像；影像數）；統計量；分母；未訓練對照；訓練動態；深度形狀；
一句話主張。「—」＝該文未報告；「n/a」＝不適用。

### 2.1 流形內在維度（非線性）

| 論文 | 網路／資料集 | 讀取張量 | 估計量 | 統計量 | 分母 | 未訓練 | 訓練動態 | 形狀 | 一句話主張 |
|---|---|---|---|---|---|---|---|---|---|
| Ansuini, Laio, Macke, Zoccolan, NeurIPS 2019 | AlexNet／VGG-11–19(±bn)／ResNet-18–152，ImageNet 預訓練；VGG-16-R 微調 1,440 張合成影像 | VGG：池化層與 FC 輸出；ResNet：**block 輸出**與最終 avgpool（攤平） | 影像為樣本；7 大類 × 500 張；末層另用 ~2,000 張 | TwoNN ID；另算 PC-ID（PC 累積變異門檻） | 無 | **有**：TwoNN 在隨機權重下「remarkably flat」；PC-ID 隨機與訓練後「qualitatively the same」 | 探索性：VGG-16／CIFAR-10 訓練中 ID 演化（Fig. 9），末層非單調 | **駝峰**（先升後降）；末層 ID 12–25 | 「the ID profile follows a typical hunchback shape」；駝峰是訓練的產物，不是 ED 初始擴張的反映 |
| Recanatesi et al., arXiv 1906.00443, 2019 | 7 層 DeepNet／Fashion-MNIST；ResNet／CIFAR-10、-100 | 各層輸出（攤平） | 影像為樣本；訓練集十類 | 局部 ID（最近鄰距離尺度）＋全域 ID（測地距離對超球比對） | 無 | **有**（Fig. 2 訓練前後） | 訓練前後對照；理論分析 SGD 壓縮 | **擴張再壓縮** | 「initial layers expand dimensionality, and final layers compress it」；ResNet／CIFAR-10 訓練「increased the global dimensionality of all the layers」 |
| Cohen, Chung, Lee, Sompolinsky, Nat. Commun. 11:746, 2020 | AlexNet／VGG-16（ResNet-50 在 SI）；ImageNet | **每一層**含 conv、ReLU 後、max-pool、FC；線性層取 ReLU 後 | 影像為樣本；點雲流形：top-10% 與全類（~1,000 張／類） | 平均場流形維度 D_M、半徑 R_M、容量；另提 spectral PR | 無 | **有**：未訓練網路「almost constant low capacity and high dimension」 | 無 | **非單調**：中段上升、末層大降（>40 → ~20） | 「pooling operations decrease both manifolds' radii and dimensions but usually increase correlations」；「dimension increases in sequences of convolutional stages without intermediate pooling」 |
| Stephenson et al., ICLR 2021 | VGG-16／AlexNet／ResNet-18，CIFAR-100（含標籤置換）；ResNet-18 寬度掃描 | 各層特徵 | 影像為樣本 | MFTMA 流形維度 D_M、半徑、中心相關 | 無 | **有**（初始化 epoch 顯示） | **有**（逐 epoch，至 1,000 epoch；回捲實驗） | 深層 D_M 在晚期下降 | 「Memorization predominately occurs in the deeper layers, due to decreasing object manifolds' radius and dimension」；寬度掃描下 D_M 呈 double descent |
| Brown, Juravsky et al., OPT-2022 workshop, arXiv 2211.13239 | ResNet-18／CIFAR-10、-100；小 transformer（grokking） | 各層激活（驗證集） | 影像為樣本 | TwoNN；LLID（末層）與 PID（峰值） | 無 | 無 | 有（transformer 逐步；ResNet 只看終點） | 駝峰；PID 恆在第一個 ResNet block 之後 | 正則化降低 LLID；PID/LLID 比值與驗證準確率相關 |
| Konz & Mazurowski, NeurIPS 2024 SciForDL workshop, arXiv 2408.08381 | VGG-13/16/19、ResNet-18/34/50；4 個自然影像＋7 個醫學影像資料集，自行訓練 | 所有池化層、conv／residual block、FC | 影像為樣本 | TwoNN | 無 | 無 | 無 | 駝峰；峰位相對深度自然 0.57、醫學 0.35 | 峰位與峰高依資料域而異；「dimensionality expansion」階段在醫學影像更早結束 |
| Valeriani et al., NeurIPS 2023（transformer，方法可移植） | iGPT-S/M/L、ESM-2；ImageNet 90,000 張 | 每個 block 第一個 norm 後，沿序列平均 | 影像為樣本 | TwoNN（Gride） | 無 | 無 | 無 | 一個顯著峰，之後平台或第二淺峰 | 「a common trait of the ID profiles is a prominent peak of the ID」；語意在第一峰末端最強 |
| Schulte & Rügamer, arXiv 2604.20276, 2026 | 重算 AlexNet／VGG／ResNet（ImageNet）逐層 TwoNN／MLE；理論 | 逐層表徵 | 影像為樣本；另分類別 | TwoNN、MLE；pointwise dimension 理論 | 無 | 無 | 無 | 實測仍是駝峰，但**理論上不可上升** | 「the ID cannot increase over the layers of any Lipschitz neural network. This theoretical result stands in stark contrast to the increasing patterns of estimated IDs」 |

### 2.2 線性秩／PCA／參與比

| 論文 | 網路／資料集 | 讀取張量 | 估計量 | 統計量 | 分母 | 未訓練 | 訓練動態 | 形狀 | 一句話主張 |
|---|---|---|---|---|---|---|---|---|---|
| Garg, Panda, Roy, IEEE Access 8, 2020 | VGG-16／AlexNet 風格，CIFAR-10、-100（ImageNet 僅 AlexNet 第一層圖） | conv 輸出，**攤平為 (N·H·W) × C**（位置為樣本） | 位置為樣本；多個 mini-batch | 達 99.9% 累積變異的 PC 數（「significant dimensions」） | 無（用來重設每層寬度） | 無 | 無 | 擴張至第 7 層後收縮（VGG-16／CIFAR-10） | 顯著維度在某層後「start contracting」即為最佳深度；ResNet 因 shortcut 未處理 |
| Elmoznino & Bonner, PLOS Comput. Biol. 20(1), 2024 | 46 個 DNN、568 個 conv 層（監督／自監督／未訓練；ImageNet、Taskonomy） | conv 層輸出，**全域平均池化**後（S6 Text：攤平 CHW 亦做） | 影像為樣本；ImageNet val 10,000 張 | 參與比 ED | 無；但以「同一層不同模型」控制 ambient 維度 | **有**：「ED is higher for trained compared with untrained models」 | 無 | **上升**：「ED tends to increase with layer depth」 | 高 ED 的模型更能預測高階視覺皮層；「feature expansion may be an important mechanism of the convolutional layers」 |
| Chen & Bonner, Sci. Adv. 11:eadw7697, 2025 | 20 個 ResNet-18 種子（Tiny ImageNet）、多架構 torchvision 集、9 個 ResNet-50 任務、20 個未訓練 ResNet-18 | 各 ReLU 層等，**全域最大池化**後 | 影像為樣本；NSD 72,128 張做 PCA，872 張評分 | PC 數取至 `torch.linalg.matrix_rank`；**未報告 ED** | 無 | **有**：未訓練網路的維度數 9,413 vs 訓練後 36,596，「due to the low-rank activation matrices of untrained networks」 | 無 | 未報告維度剖面；普適性「不隨層強烈變化」 | 「diverse networks learn to represent natural images using a shared set of latent dimensions」；未訓練網路的共享維度來自影像統計 |
| Kong et al., PLOS Comput. Biol. 18(1), 2022 | ResNet-50 robust vs non-robust；ImageNet val 子集 | 各層激活 | 影像為樣本 | 特徵值譜冪律指數 α | 無 | 無 | 無 | 未報告形狀；robust 模型 α 略高 | 對抗訓練後 α 更接近 V1 |
| Ghosh, Mondal, Agrawal, Richards, arXiv 2202.05808, 2022 | VGG／ResNet／ViT，ImageNet 監督與 SSL | 中間層特徵 | 影像為樣本 | 冪律指數 α | 無 | 無 | 無 | 深層 α → 1 | 「Representations from deeper layers exhibit α closer to 1 as compared to intermediate layer representations」 |
| Agrawal, Mondal, Ghosh, Richards, NeurIPS 35:17626–17638, 2022（α-ReQ） | VGG-13/16/19（dropout 與 MaxPool 的**輸入**）、ResNet-13/50/101（每個 residual block 的**輸入**＋avgpool）、ViT-B/8、L/16、H/14；ImageNet；ResNet-50 SimCLR／BYOL／BarlowTwins | 見左；conv 讀 pre-pool | 影像為樣本 | 共變異特徵值譜 log-log 線性擬合的 α | 無 | 無 | 僅 BarlowTwins 模型選擇 | CNN 中間層 α < 1，最深層 α ≈ 1；ViT 早層 α > 1 | 「representations extracted from the deepest layers of all the models exhibit α close to 1, irrespective of the total depth of each model」 |
| Huh et al., TMLR 2023（low-rank simplicity bias） | MLP／小 CNN／AlexNet／ResNet-10/18；CIFAR-10/100、ImageNet | **倒數第二層**嵌入的 Gram 矩陣（cosine kernel，測試集） | 影像為樣本 | Roy–Vetterli 有效秩（對 Gram 矩陣） | 無 | **有**：「Randomly initialized deep nets are biased to correspond to Gram matrices with a low effective rank」 | **有**：CNN／CIFAR-100「first exhibit effective rank contracting behavior throughout training, and then the effective rank starts to increase again」 | 對**網路深度**遞減，非逐層剖面 | 越深的網路找到越低有效秩的嵌入，初始化與收斂後皆然 |
| Zhou et al., arXiv 2606.18676, 2026（InTrain） | NAS-Bench 候選網路（CIFAR-10/100、ImageNet16-120），**未訓練** | 各層激活 A_ℓ ∈ R^{N×C}（空間展開） | 影像為樣本 | 各層激活共變異的參與比 | 無 | 全為未訓練（零成本 NAS 代理） | 無 | 未報告 | 逐層 PR 之幾何容量 × 梯度韌性可預測可訓練性 |
| Smirnov et al., arXiv 2607.19315, 2026（ERank in latent space） | ResNet-18（layer1–4）、CLIP ViT-B/32；ImageNet 預訓練，凍結 | 各層特徵圖攤平為 **HW × C**（**位置為樣本，單張影像**） | 位置為樣本；每張影像獨立 | Roy–Vetterli 有效秩（通道共變異） | 隱含 min(N, C) 上界 | 無 | 無 | 未報告剖面（每層獨立算） | 「ERank counts how many decorrelated channel directions an image activates」；作為影像豐富度指標 |
| Chun, Canatar, Chung, Lee, ICLR 2026 | 神經記錄（鈣成像、電生理、fMRI）＋ LLM 各層 | LLM 各層 token 特徵 | 有限樣本偏差修正 | 參與比及其無偏估計；局部維度 | 無 | 無 | 無 | LLM 逐層剖面（偏差修正後更平） | 「the participation ratio of eigenvalues ... is highly biased with small sample sizes」 |

### 2.3 神經崩塌與 tunnel effect（數值秩）

| 論文 | 網路／資料集 | 讀取張量 | 估計量 | 統計量 | 分母 | 未訓練 | 訓練動態 | 形狀 | 一句話主張 |
|---|---|---|---|---|---|---|---|---|---|
| Papyan, Han, Donoho, PNAS 117(40), 2020 | VGG／ResNet／DenseNet；MNIST…ImageNet | **僅末層**（分類器前） | 影像為樣本 | NC1–NC4（類內變異、ETF、對齊） | n/a | 無 | **有**（TPT） | n/a | 末層類均值收斂為 simplex ETF、類內變異塌縮 |
| Rangamani, Lindegaard, Galanti, Poggio, ICML 2023（PMLR 202:28729–28745） | MLP／ConvNet／ResNet-18/34/50；MNIST、FashionMNIST、CIFAR-10、SVHN | 逐層特徵，**以類均值置中**的類內矩陣 H_c^ℓ | 影像為樣本 | 穩定秩 ‖H‖_F²／‖H‖_2² | 無 | 僅作 NC3 對齊基線 | **有**（350 epoch） | 「first increases and then decreases in the collapsed layers」 | 「as we move deeper into a trained neural network, the within-class covariance decreases relative to the between-class covariance」 |
| Masarczyk et al., NeurIPS 2023（tunnel effect） | MLP（1024 寬，6–12 層）、VGG-16/19、ResNet-18/34（**只讀 conv2**）；CIFAR-10/100、CINIC-10 | 攤平表徵，「randomly choose a subset of 8000 features」 | 影像為樣本（張數未明） | 樣本共變異奇異值 > 1e-3·σ_1 的個數（數值秩） | 無 | 無明確對照（但 Fig. 6 從第 0 步畫起） | **有**：前 75 步逐步；之後每 10 epoch | **崩到接近類別數**；ResNet-34 在第 29 層有尖峰 | 「the rank collapses to values near-the-number of classes. It stays in this regime until the end of the training」；tunnel 長度隨 VGG 寬度增加，ResNet-34 不然 |
| Harun, Lee, Gallardo, Krishnan, Kanan, NeurIPS 2024 | 64 個自訓 DNN（VGGm-11/17、VGGm†、ResNet-18/34、ViT-T+；ImageNet-100 為主，CIFAR）＋ 8 個 ImageNet-1k 預訓練（ResNet-50 SL/SSL、ViT-B、ConvNeXt-B） | 逐層線性探針；附錄 Fig. 25 數值秩（沿用 [Masarczyk]） | 影像為樣本 | 數值秩（附錄）；主體是 OOD 探針指標 | 無 | 無 | 無（只看終點） | 32×32 訓練時秩在 extractor 後「plummets」；224×224「retains a much higher rank」 | 「this is not a universal phenomenon」；去掉前兩個 stage 的 max-pool（VGGm†）即「eliminates the tunnel」；ImageNet-1k 預訓練模型中只有 ResNet-50 有 tunnel |
| Masarczyk et al., arXiv 2506.01562, 2025（softmax temperature） | MLP／VGG／ResNet-18/20/34/50／ViT；CIFAR-10/100、ImageNet-100/1k | logits 矩陣；MLP 逐層激活 A_i（Fig. 4） | 影像為樣本 | 數值秩 | 無 | 無 | **有** | 深層激活秩低於類別數（rank-deficit） | 「rank(∂L/∂W_i) ≤ rank(A_{i−1})」，表徵塌縮限制梯度多樣性 |
| Kubaty et al., arXiv 2407.14320, 2025（multi-exit） | ResNet-20/34/50、ViT-T/B；CIFAR-100、ImageNet-1k、TinyImageNet | 骨幹各層激活矩陣（n 樣本 × m 特徵） | 影像為樣本 | 數值秩 | 無 | 無 | 三種訓練制度比較 | 「a higher rank in earlier layers and a lower rank in deeper layers」 | 讓早退分類器影響骨幹會把深層秩抬高 |

### 2.4 SSL 表徵秩指標（單層，非剖面）

| 論文 | 網路／資料集 | 讀取張量 | 估計量 | 統計量 | 分母 | 未訓練 | 訓練動態 | 形狀 | 一句話主張 |
|---|---|---|---|---|---|---|---|---|---|
| Garrido, Balestriero, Najman, LeCun, ICML 2023（PMLR 202:10929–10974，RankMe） | ResNet-50 骨幹＋MLP 投影器；SimCLR／VICReg／DINO 等；ImageNet-1k | 嵌入（投影器輸出）與表徵（骨幹輸出），**單層** | 影像為樣本；「we use 25600 samples」 | exp(Shannon 熵) 有效秩 | 以嵌入維度 2048 為上限：「the dimension of the manifold ... cannot be higher than 2048 ... we clip the value」 | 無 | 跨超參數，非逐 epoch | n/a | 「having a high rank is a necessary ... condition」；可無標籤選超參數 |
| Thilak et al., ICLR 2024（LiDAR） | ViT-B（I-JEPA、data2vec）、ViT-S DINO、ResNet-50 VICReg；ImageNet-1k | 嵌入，單層 | 影像為樣本（n 個乾淨樣本 × q 個增強） | LDA 矩陣 Σ_w^{-1}Σ_b 的有效秩 | Σ_b 秩 ≤ n | 無 | 檢查點 epoch 20–100 | n/a | RankMe 把無資訊方向也算進秩；LiDAR「discriminating between informative and uninformative features」 |
| Zhuo, Wang, Ma, Wang, ICLR 2023（rank differential） | BYOL／SimSiam／SwAV／DINO，ResNet，CIFAR-10/100、ImageNet-100 | online 與 target 分支輸出 | 影像為樣本 | 有效秩（特徵相關矩陣） | 無 | 無 | **有**（逐 epoch） | n/a | 非對稱設計造成 erank(C_p) < erank(C_z)，藉此避免維度塌縮 |
| Jing, Vincent, LeCun, Tian, ICLR 2022（dimensional collapse） | SimCLR ResNet-50 ImageNet | 嵌入的共變異奇異值譜 | 影像為樣本 | 奇異值譜（無單一統計量） | 無 | 無 | 無 | n/a | 對比式 SSL 也有維度塌縮，源於強增強與隱式正則化 |

### 2.5 剪枝文獻中的激活秩

| 論文 | 網路／資料集 | 讀取張量 | 估計量 | 統計量 | 分母 | 未訓練 | 訓練動態 | 形狀 | 一句話主張 |
|---|---|---|---|---|---|---|---|---|---|
| Lin et al., CVPR 2020（HRank） | VGG-16、ResNet-56/110、DenseNet-40、GoogLeNet、ResNet-50；CIFAR-10、ImageNet | **單一通道**的 H×W 特徵圖（conv 後） | 每通道對多張影像取平均 | 單一特徵圖的 SVD 秩 | 隱含 min(H, W) | 無 | 無 | 逐層報告每通道秩，不是通道空間秩 | 單通道秩對 batch 穩定，可當剪枝準則。**不是通道共變異譜**，常被誤引 |
| Sui et al., NeurIPS 2021（CHIP） | ResNet-50 等；CIFAR-10、ImageNet | 一層全部特徵圖矩陣化 | 逐影像 | 移除一張特徵圖後**核範數**的變化（channel independence） | 無 | 無 | 無 | 未報告 | 「nuclear norm change better reveals the impact of deleted feature maps」，秩太硬 |

### 2.6 秩隨深度的理論與 transformer（方法可移植）

| 論文 | 網路／資料集 | 讀取張量 | 估計量 | 統計量 | 分母 | 未訓練 | 訓練動態 | 形狀 | 一句話主張 |
|---|---|---|---|---|---|---|---|---|---|
| Daneshmand, Kohler, Bach, Hofmann, Lucchi, NeurIPS 2020 | ReLU MLP（深 1–32、寬 128）Fashion-MNIST；VGG-19、ResNet-50 CIFAR-10 | **末隱藏層**激活對 batch 的矩陣 | batch 為樣本 | 軟秩 rank_τ（σ_i²/N ≥ τ）；`torch.matrix_rank` | 寬度 d（BN 保 Ω(√d)） | 未訓練即是對象 | **有**（rank 隨 epoch） | 秩對**網路深度**塌縮（無 BN）；非逐層剖面 | 「SGD updates preserve the order of the initial rank throughout optimization」 |
| Feng et al., NeurIPS 2022（rank diminishing） | ResNet-18/50、GluMixer-24、ResMLP-S24、Swin-T、ViT-T；ImageNet | **Jacobian** 的部分秩（逐層）；最終特徵的「classification dimension」 | 影像為樣本 | 數值秩；保留 95% 分類準確率所需的 PC 數 | 無 | 無 | 無 | 單調遞減 | 「the rank of the network Jacobian and intrinsic dimension of feature manifolds decrease monotonically by depth」 |
| Baker et al., arXiv 2402.06751, 2024 | FC／RNN／ResNet16／VGG11；MNIST、CIFAR-10、TinyImageNet | **梯度**、激活、delta 的秩，逐層、逐 epoch | batch 為樣本 | 數值秩（機器精度門檻） | 無 | 有（訓練起點） | **有** | 瓶頸層後梯度秩受激活秩限制 | 梯度秩 ≤ min(rank Z_{i−1}, rank Δ_i)；卷積 stride／kernel 顯式界 |
| Makwana, arXiv 2507.07675, 2025 | 隨機高斯 ReLU MLP（理論） | m×n 隱藏激活矩陣 | 固定 m 個輸入 | 期望秩 E[EDim(ℓ)] | m | 全為未訓練 | 無 | 幾何衰減後**振盪**（revival depths） | 「rank deficit decays geometrically with ratio 1 − 2/π」；正交初始化或 leaky-ReLU 則近滿秩 |
| Everett, arXiv 2607.14018, 2026（Transforming Rank） | 殘差 ReLU MLP／Transformer FFN block（初始化）；CIFAR-10 可訓練性 | 輸入–輸出 **Jacobian** | 12 個種子 | Roy–Vetterli 有效秩，**除以寬度 d** | **標稱寬度 d** | 全為初始化 | 無（僅預測可訓練性） | 無 skip 隨深度塌縮；Pre-Norm 平台 | 「skip connections route the gradient around the residual branch, where rank is lost」；4× 寬度擴張使分支 Jacobian 滿秩，門檻循 Marchenko–Pastur |
| Jha & Reagen, arXiv 2510.00537, 2025（Spectral scaling laws） | LLaMA-130M/250M、GPT-2、nGPT | FFN **post-activation** 共變異，逐層 | token 為樣本 | Hard Rank（參與比）、Soft Rank（Shannon）、**除以 D 的利用率** | **FFN 隱藏寬度 D** | 無 | **有** | 逐層熱圖；利用率隨寬度下降 | 秩對寬度次線性成長，「recast width selection as a spectral utilization optimization problem」 |
| Skean et al., ICML 2025（Layer by Layer） | LLM；ViT／CLIP／BEiT／AIM／DINO 等視覺 transformer | 各層 token 表徵 | token／影像為樣本 | 矩陣熵（統一 erank、curvature、InfoNCE、LiDAR） | 無 | 無 | 微調前後 | 中層壓縮，中層探針最佳 | 中間層表徵常優於末層 |

---

## 3. 尚未引用、必須補的

**A 級（不引會被審稿人直接點名）**

1. `masarczyk2023tunnel` —— 唯一在 CNN 上做逐層數值秩 × 訓練動態的 NeurIPS 論文；
   我們的「epoch 40 定型」、「深層秩塌」、ResNet vs VGG 差異都要對照它。
2. `harun2024tunnelVariables` —— 2024 NeurIPS，直接測 max-pool 對 tunnel 的作用、
   stem 大小的作用，且指出 ImageNet-1k 預訓練模型中只有 ResNet-50 有 tunnel。
   我們六個 ResNet-50 配方的結果要跟這句對話。
3. `cohen2020manifolds` —— 池化降維的流形版先例；也是「未訓練網路剖面平坦」的
   先例之一（與 Ansuini 並列）。
4. `garrido2023rankme`、`agrawal2022alphaReq` —— SSL 秩指標；α-ReQ 有 VGG／ResNet
   逐層 α 剖面（讀 pre-pool 輸入），是我們之外少數在 ImageNet 預訓練 CNN 上做
   逐層譜統計的。
5. `huh2023lowRank` —— 「初始化即低秩」與訓練中有效秩非單調，牽涉承重主張 4、7。
6. `chen2025universal` —— 20 個 ResNet-18 種子＋20 個未訓練種子，正是規則 10 說
   我們缺的東西；且明說未訓練激活矩陣低秩。
7. `schulte2026rethinking` —— 見結論 3。
8. `chun2026estimating` —— PR 的有限樣本偏差；我們 n/C ≥ 50 的門檻要引它。
9. `smirnov2026erank` —— 唯一同樣以**位置為樣本**、算通道共變異有效秩的工作
   （單張影像）；我們的估計量要與它區隔（我們跨影像、跨位置，且有分母）。

**B 級（相關工作段落應提）**

`thilak2024lidar`、`konz2024preprocessing`、`stephenson2021geometry`、
`valeriani2023geometry`、`kubaty2025multiExit`、`masarczyk2025softmax`、
`jha2025spectralScaling`、`everett2026transformingRank`、`baker2024gradientRank`
（`rmax-novelty` 表裡已列但 refs.bib 沒有）、`sui2021chip`、`brown2022regularization`、
`zhou2026intrain`、`makwana2025oscillations`。

**C 級（一句話帶過或不引）**

`zhuo2023rankDifferential`、`jing2022dimensionalCollapse`、`skean2025layerByLayer`、
`sukenik2024ncLowRank`、`galanti2022minimalDepth`、`benshaul2022ncc`、
`he2023separation`、`doimo2020nucleation`、`zhang2026residual`、`wen2025cbn`。

---

## 4. 與承重主張的對照：誰先講過、誰撞到

### 4.1 r_max 是對的分母

- **無人用可達秩當分母。** 用**標稱寬度**當分母的：Everett 2026（erank(∂h_L/∂h_0)/d）、
  Jha & Reagen 2025（rank/D 稱「utilization」，LLM FFN）、Garrido 2023（上限 2048）。
  Elmoznino 2024 不正規化，改以「同一層不同模型」控制 ambient 維度。Smirnov 2026
  只寫「bounded between 1 and min(N, C)」。
- **最接近的先例仍是 Han 2021／Kim 2018**（已引）。新增：Baker 2024 為**梯度**秩寫出
  含 stride／kernel 的顯式界；Wen & Jacot 2025 定義 CNN 的「Convolution Bottleneck
  rank」（理論）。兩者都不是激活量測。
- **可能被拿來對照的觀察**：Masarczyk 2023 在 ResNet-34 第 29 層看到秩尖峰，
  「coincides with the end of the penultimate residual stage」，未解釋；Harun 2024
  發現 7×7 stem（ResNet-18）比 3×3 stem（VGGm-17）OOD 差。這兩件事我們的
  r_max 框架都可以量，但論文裡只能寫「可檢驗」，不能寫「解釋了」。
- **措辭警戒**：仍照 `rmax-novelty` §4——「文獻忽略了一個已知上界」。

### 4.2 池化一致壓低水準

- **直接先例（必引、必區隔）**：Cohen et al. 2020 Fig. 7b：「pooling operations
  decrease both manifolds' radii and dimensions but usually increase correlations
  (figure 7b), the latter is presumably due to the underlying spatial averaging」；
  Fig. 5a 的非單調性「where dimension increases in sequences of convolutional
  stages without intermediate pooling」。這是**平均場流形維度**，AlexNet／VGG-16，
  ImageNet；不是線性秩，沒有分母，也沒有 687/703 這種逐層配對統計。
- **機制上的先例**：Harun 2024 §4.1.4 spatial reduction ϕ：VGGm（五個 stage 都有
  max-pool）有 tunnel，VGGm†（前兩個 stage 去掉 max-pool）「eliminates the tunnel
  for all OOD datasets」，OOD 保留率 64.85% → 84.40%（p < 0.001）。他們量的是探針
  與（附錄）數值秩。
- **表面上的反例**：Ansuini 2019 VGG-16-R「the ID first increased in the first
  pooling layer」——TwoNN ID 在 pool1 上升。這是相對於輸入影像、且是流形 ID；
  我們的主張是「池化層 vs 其前一個 conv 層」的線性秩水準。寫的時候要把這句
  點出來，免得被當成矛盾。α-ReQ 讀的是 MaxPool 的**輸入**，無法比較。
- `notes/pooling-novelty-2026-09-03.md` 若沒有 Cohen 2020 與 Harun 2024，要補。

### 4.3 block 輸出超過 conv3 上界、比值約 4

- 無人同時量 conv 輸出與 block 輸出。Ansuini／Elmoznino 讀 block 輸出；Masarczyk
  只讀 conv2；α-ReQ 讀 block **輸入**（＝前一個 block 輸出）。
- **方向一致的旁證**：Zhang et al. 2026（arXiv 2404.10947）主張恆等捷徑「injecting
  an echo of shallower representations into deeper layers」，把捷徑衰減後特徵有效
  秩降低且探針更好（MAE ViT）；Everett 2026：「Skip connections restore the effective
  rank that each feedforward block contracts」（Jacobian，初始化）。都不矛盾。
- Harun 2024：ImageNet-1k 預訓練的 ResNet-50（SL 與 SSL）是唯一「weakly present」
  tunnel 的模型，ConvNeXt-B、ViT-B 沒有——與我們「bottleneck 家族行為不同」
  的方向一致，但他們沒把原因歸到 1×1 擴張。

### 4.4 訓練後順序：bottleneck 家族學出來、plain／basic／dense 由架構決定

- **支持 plain 家族「架構決定」**：Ansuini 2019 Fig. 5C：VGG-16 的 **PC-ID**（線性
  計數，與我們的 k* 同類）在隨機權重下「its profile was qualitatively the same as
  in trained networks」，只有 TwoNN ID 才變平。這是我們 T-R 0.6–0.75 的直接先例，
  必引。
- **未訓練水準較低**：Elmoznino 2024（ED 訓練後較高）、Chen & Bonner 2025（未訓練
  維度數約四分之一）、Recanatesi 2019（ResNet／CIFAR-10 訓練「increased the global
  dimensionality of all the layers」）、Huh 2023（初始化即低秩）。與「水準抬高」一致。
- **潛在張力**：Daneshmand 2020「SGD updates preserve the order of the initial rank
  throughout optimization」——若讀成「初始秩序保持」，會被拿來質疑 bottleneck
  家族「順序學出來」（T-R −0.08）。但他們量的是**末隱藏層**秩對**網路深度**，
  MLP／VGG-19，不是逐層剖面。要在文中先講清楚。
- **沒有人區分 block 類型。** 這一列的新穎性成立，但規則 10 的問題（非 ResNet-50
  家族無誤差棒）在 Chen & Bonner 有 20 個種子的對照下會更刺眼。

### 4.5 訓練把瓶頸從資料移到 kernel

- 先例都在隨機網路：Han 2021（已引）、Daneshmand 2020、Makwana 2025（隨機 ReLU
  層秩虧損 geometric decay ＋ revival）、Everett 2026。無人在訓練後 ImageNet 網路上
  量「哪一側咬住」。不矛盾。

### 4.6 k*(0.999) 的功能性校準

- **直接先例**：Feng 2022 §3.3「classification dimension」＝保留 95% 分類準確率所需
  的最少 PC 數（ResNet-18/50、Swin、ViT 末層特徵）。這就是功能性校準，只做末層。
  我們的 k*(0.999) 逐層校準要引它並區隔。
- **門檻家族**：Garg 99.9%（已引）；Masarczyk 1e-3·σ_1；Daneshmand rank_τ；
  Chen & Bonner `matrix_rank` 預設；Baker 機器精度。論文寫 τ 的選擇時要列這串。
- **估計量偏差**：Chun 2026（PR 小樣本高估／低估）、Pospisil 2025（已引）。
  我們的 n/C ≥ 50 門檻要同時引。

### 4.7 剖面在 epoch 40 定型（plain VGG／CIFAR）

- **先例（更早、更強）**：Masarczyk 2023 Fig. 6：VGG-19／CIFAR-10「the rank
  collapses to values near-the-number of classes」在**前 75 個訓練步**內發生，
  「It stays in this regime until the end of the training」；Fig. 5「Early in
  training, tunnel layers stabilize」，LR 在 epoch 80、120 衰減。我們 §9 的
  ρ(final, init) = 0.93、epoch 40 定型，是同一現象的線性秩／r_max 版本，必須引
  且說明我們多了什麼（正規化、六配方、全層配對檢定）。
- **反方向的證據（要寫進限制）**：Huh 2023 小 CNN／CIFAR-100「first exhibit
  effective rank contracting behavior throughout training, and then the effective
  rank starts to increase again」；Stephenson 2021 深層流形維度在**晚期** epoch
  因記憶化（含標籤雜訊）下降；Ansuini 2019 末層 ID「non-monotonic variation」；
  Rangamani 2023 穩定秩在 350 epoch 內隨 LR 衰減仍在變。所以「定型」只能寫成
  「在此 LR 排程下、以 Spearman 順序衡量」，不能寫成秩不再變。

### 4.8 對規則 4 的補充（四種維度分開）

- Schulte & Rügamer 2026 的定理只約束流形 ID（pointwise dimension），與線性秩無關；
  非線性抬高線性秩是 Han 2021 §3.2 明講的。論文寫「上升」時要註明是線性計數。
- Cohen 2020 也明講 ReLU「increases manifolds' dimensions」而 pooling 降低——
  同一層內兩個操作方向相反，是我們「conv 與 post-activation 分開記」（規則 6）的
  流形版理由。

---

## 5. LaTeX 比較表草稿與 BibTeX

### 5.1 table*（booktabs）

只放 CNN 逐層剖面與方法直接可比的列；SSL 單層指標與 transformer 放註腳或正文。
鍵值依 refs.bib 慣例 surnameYYYYkeyword。

```latex
\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{Per-layer dimensionality profiles of CNN representations in prior work.
Tensor: what is read at each layer; Samples: what plays the role of a sample;
Denom.: normaliser applied to the statistic; Untr.: untrained control; Dyn.:
training dynamics; Shape: reported profile along depth. ``Lin.\ rank'' groups
numerical/stable/effective rank of the activation (or centred activation)
matrix; ``ID'' is a manifold intrinsic-dimension estimator.}
\label{tab:profile_survey}
\begin{tabular}{@{}llllllccl@{}}
\toprule
Work & Networks / data & Tensor & Samples & Statistic & Denom. & Untr. & Dyn. & Shape \\
\midrule
\multicolumn{9}{@{}l}{\emph{Manifold intrinsic dimension}} \\
\cite{ansuini2019intrinsic} & AlexNet, VGG, ResNet; ImageNet & block / pool out & images (7$\times$500) & TwoNN ID, PC-ID & -- & \checkmark & (\checkmark) & hunchback \\
\cite{recanatesi2019dimensionality} & MLP, ResNet; F-MNIST, CIFAR & layer out & images & local/global ID & -- & \checkmark & before/after & expand--compress \\
\cite{cohen2020manifolds} & AlexNet, VGG-16, ResNet-50; ImageNet & every op.\ incl.\ pool & images (class clouds) & MFT manifold dim.\ $D_M$ & -- & \checkmark & -- & non-monotone; pool $\downarrow$ \\
\cite{stephenson2021geometry} & VGG-16, ResNet-18; CIFAR-100 & layer out & images & MFT $D_M$ & -- & \checkmark & \checkmark & deep $\downarrow$ late \\
\cite{konz2024preprocessing} & VGG-13/16/19, ResNet-18/34/50; 11 datasets & pool / block / FC & images & TwoNN ID & -- & -- & -- & hunchback (peak 0.57 vs 0.35) \\
\cite{schulte2026rethinking} & AlexNet, VGG, ResNet; ImageNet & layer out & images & TwoNN, MLE (+theory) & -- & -- & -- & cannot rise (theory) \\
\midrule
\multicolumn{9}{@{}l}{\emph{Linear rank / PCA / participation ratio}} \\
\cite{garg2019lowEffort} & VGG/AlexNet-style; CIFAR & conv out, flattened & positions & \#PC at 99.9\% var. & -- & -- & -- & expand to layer 7, then contract \\
\cite{elmoznino2024highDim} & 46 DNNs, 568 conv layers & conv out, GAP & images (10k) & PR & -- & \checkmark & -- & rising \\
\cite{chen2025universal} & ResNet-18 $\times$20 seeds, ResNet-50 $\times$9 tasks & ReLU out, GMP & images (72k) & rank (matrix\_rank) & -- & \checkmark & -- & not reported \\
\cite{agrawal2022alphaReq} & VGG-13/16/19, ResNet-50/101, ViT; ImageNet & pre-pool / block in & images & power-law $\alpha$ & -- & -- & -- & $\alpha<1$ mid, $\to1$ deep \\
\cite{huh2023lowRank} & MLP, CNN, ResNet; CIFAR, ImageNet & penultimate Gram & images & erank & -- & \checkmark & \checkmark & vs.\ net depth $\downarrow$ \\
\cite{smirnov2026erank} & ResNet-18 layers 1--4; ImageNet & feature map $HW\times C$ & positions (1 image) & erank & $\min(N,C)$ & -- & -- & per-layer, no profile \\
\midrule
\multicolumn{9}{@{}l}{\emph{Neural collapse / tunnel effect}} \\
\cite{rangamani2023intermediate} & MLP, ConvNet, ResNet-18/34/50; MNIST, CIFAR-10, SVHN & centred within-class feat. & images & stable rank & -- & -- & \checkmark & rise then fall \\
\cite{masarczyk2023tunnel} & MLP, VGG-19, ResNet-34 (conv2); CIFAR, CINIC & flattened (8k feats) & images & num.\ rank ($10^{-3}\sigma_1$) & -- & -- & \checkmark & collapse to \#classes \\
\cite{harun2024tunnelVariables} & VGGm, ResNet, ViT; ImageNet-100 & flattened & images & num.\ rank (app.) & -- & -- & -- & collapse at $32^2$, kept at $224^2$ \\
\cite{kubaty2025multiExit} & ResNet-20/34/50, ViT; CIFAR-100, ImageNet & activation matrix & images & num.\ rank & -- & -- & regimes & high early, low deep \\
\midrule
\multicolumn{9}{@{}l}{\emph{Other objects (for contrast)}} \\
\cite{feng2022rankDiminishing} & ResNet-18/50, Swin, ViT; ImageNet & \emph{Jacobian} & images & partial num.\ rank & -- & -- & -- & monotone $\downarrow$ \\
\cite{lin2020hrank} & VGG-16, ResNet-50; CIFAR, ImageNet & single $H\times W$ map & per channel & SVD rank & $\min(H,W)$ & -- & -- & per-channel \\
\cite{daneshmand2020bn} & MLP; VGG-19, ResNet-50; CIFAR-10 & last hidden layer & batch & soft rank & width $d$ & \checkmark & \checkmark & vs.\ net depth \\
This work & 22 checkpoints, 7 families; ImageNet & conv out, pre-act., pre-residual & positions $\times$ images & $k^*(0.999)$ & $r_{\max}$ & \checkmark & \checkmark & see \S V \\
\bottomrule
\end{tabular}
\end{table*}
```

### 5.2 BibTeX（新條目；DOI 皆經 Crossref 查證，無 DOI 者標 UNVERIFIED）

```bibtex
% ===== Per-layer profiles: tunnel effect and neural collapse ==============

@inproceedings{masarczyk2023tunnel,
  author    = {Masarczyk, Wojciech and Ostaszewski, Mateusz and Imani, Ehsan and
               Pascanu, Razvan and Mi{\l}o{\'s}, Piotr and Trzci{\'n}ski, Tomasz},
  title     = {The Tunnel Effect: Building Data Representations in Deep Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  year      = {2023},
  doi       = {10.52202/075280-3355},
  annote    = {arXiv:2305.19753. Numerical rank (sigma > 1e-3 sigma_1) of a random
               8000-feature subset, flattened; ResNet-34 read at conv2 only.
               Rank collapses to ~#classes within the first 75 training steps and
               stays there (Fig. 6). Tunnel length grows with VGG width, not ResNet},
}

@inproceedings{harun2024tunnelVariables,
  author    = {Harun, Md Yousuf and Lee, Kyungbok and Gallardo, Jhair and
               Krishnan, Giri and Kanan, Christopher},
  title     = {What Variables Affect Out-of-Distribution Generalization in
               Pretrained Models?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  doi       = {10.52202/079017-1799},
  annote    = {arXiv:2405.15018. 64 DNNs, 8,604 linear probes. Removing max-pool
               from the first two stages (VGGm-dagger) eliminates the tunnel;
               larger stems hurt OOD; among ImageNet-1k pretrained models only
               ResNet-50 (SL and SSL) shows a tunnel. Numerical rank in App. Fig. 25},
}

@article{masarczyk2025softmax,
  author  = {Masarczyk, Wojciech and Ostaszewski, Mateusz and Cheng, Tin Sum and
             Trzci{\'n}ski, Tomasz and Lucchi, Aurelien and Pascanu, Razvan},
  title   = {Unpacking Softmax: How Temperature Drives Representation Collapse,
             Compression and Generalization},
  journal = {arXiv preprint arXiv:2506.01562},
  year    = {2025},
  annote  = {Rank-deficit bias: numerical rank of logits/activations below the
             number of classes; rank(dL/dW_i) <= rank(A_{i-1})},
}

@article{kubaty2025multiExit,
  author  = {Kubaty, Piotr and W{\'o}jcik, Bartosz and Krzepkowski, Bart{\l}omiej and
             Michaluk, Monika and Trzci{\'n}ski, Tomasz and Pomponi, Jary and
             Adamczewski, Kamil},
  title   = {How to Train Your Multi-Exit Model? Analyzing the Impact of Training
             Strategies},
  journal = {arXiv preprint arXiv:2407.14320},
  year    = {2025},
  annote  = {Numerical rank of backbone activation matrices per layer; higher in
             early layers, lower in deep layers},
}
% UNVERIFIED: peer-reviewed venue (v2 June 2025) not identified.

% ===== Manifold intrinsic dimension ======================================

@article{cohen2020manifolds,
  author  = {Cohen, Uri and Chung, SueYeon and Lee, Daniel D. and Sompolinsky, Haim},
  title   = {Separability and Geometry of Object Manifolds in Deep Neural Networks},
  journal = {Nature Communications},
  volume  = {11},
  pages   = {746},
  year    = {2020},
  doi     = {10.1038/s41467-020-14578-5},
  annote  = {Mean-field manifold dimension per operation in AlexNet/VGG-16
             (ResNet-50 in SI), ImageNet. Untrained: constant high dimension.
             Fig. 7b: pooling decreases manifold dimension; ReLU increases it},
}

@inproceedings{stephenson2021geometry,
  author    = {Stephenson, Cory and Padhy, Suchismita and Ganesh, Abhinav and
               Hui, Yue and Tang, Hanlin and Chung, SueYeon},
  title     = {On the Geometry of Generalization and Memorization in Deep Neural
               Networks},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  annote    = {arXiv:2105.14602. Manifold dimension per layer and per epoch in
               VGG-16/CIFAR-100; memorization changes deep-layer dimension late
               in training; dimension shows double descent across ResNet-18 width},
}
% UNVERIFIED: no DOI (ICLR).

@inproceedings{konz2024preprocessing,
  author    = {Konz, Nicholas and Mazurowski, Maciej A.},
  title     = {Pre-processing and Compression: Understanding Hidden Representation
               Refinement Across Imaging Domains via Intrinsic Dimension},
  booktitle = {NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep
               Learning (SciForDL)},
  year      = {2024},
  annote    = {arXiv:2408.08381. TwoNN per layer for VGG-13/16/19 and
               ResNet-18/34/50 on 4 natural + 7 medical datasets; hunchback peak
               at relative depth 0.57 (natural) vs 0.35 (medical)},
}
% UNVERIFIED: no DOI (workshop).

@inproceedings{brown2022regularization,
  author    = {Brown, Bradley C. A. and Juravsky, Jordan and Caterini, Anthony L.
               and Loaiza-Ganem, Gabriel},
  title     = {Relating Regularization and Generalization through the Intrinsic
               Dimension of Activations},
  booktitle = {NeurIPS 2022 Workshop on Optimization for Machine Learning (OPT)},
  year      = {2022},
  annote    = {arXiv:2211.13239. TwoNN per layer for ResNet-18/CIFAR; peak ID
               after the first ResNet block},
}
% UNVERIFIED: full author list taken from the arXiv listing; no DOI.

@inproceedings{valeriani2023geometry,
  author    = {Valeriani, Lucrezia and Doimo, Diego and Cuturello, Francesca and
               Laio, Alessandro and Ansuini, Alessio and Cazzaniga, Alberto},
  title     = {The Geometry of Hidden Representations of Large Transformer Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  year      = {2023},
  doi       = {10.52202/075280-2230},
  annote    = {arXiv:2302.00294. TwoNN profile of iGPT/ESM-2: one prominent peak
               then plateau or second shallow peak},
}

@article{schulte2026rethinking,
  author  = {Schulte, Rickmer and R{\"u}gamer, David},
  title   = {Rethinking Intrinsic Dimension Estimation in Neural Representations},
  journal = {arXiv preprint arXiv:2604.20276},
  year    = {2026},
  annote  = {Theorem: the ID cannot increase across layers of a Lipschitz network;
             rising estimated-ID patterns are estimator artefacts. Applies to
             manifold ID, not to linear rank},
}

% ===== Linear rank / PR / spectra of representations =====================

@article{huh2023lowRank,
  author  = {Huh, Minyoung and Mobahi, Hossein and Zhang, Richard and Cheung, Brian
             and Agrawal, Pulkit and Isola, Phillip},
  title   = {The Low-Rank Simplicity Bias in Deep Networks},
  journal = {Transactions on Machine Learning Research},
  year    = {2023},
  annote  = {arXiv:2103.10427. Effective rank of the penultimate-layer Gram matrix
             (cosine kernel). Low rank already at initialisation; during training
             on CIFAR-100 the effective rank contracts and then rises again},
}
% UNVERIFIED: TMLR has no DOI; month (3/2023) from the PDF header.

@article{chen2025universal,
  author  = {Chen, Zirui and Bonner, Michael F.},
  title   = {Universal Dimensions of Visual Representation},
  journal = {Science Advances},
  volume  = {11},
  pages   = {eadw7697},
  year    = {2025},
  doi     = {10.1126/sciadv.adw7697},
  annote  = {arXiv:2408.12804. Global max-pooled ReLU outputs, PCA on 72,128 NSD
             images, PCs up to matrix_rank. 20 trained vs 20 untrained ResNet-18
             seeds: 36,596 vs 9,413 dimensions -- untrained activation matrices
             are low-rank. No ED profile reported},
}

@inproceedings{agrawal2022alphaReq,
  author    = {Agrawal, Kumar Krishna and Mondal, Arnab Kumar and Ghosh, Arna and
               Richards, Blake A.},
  title     = {$\alpha$-{ReQ}: Assessing Representation Quality in Self-Supervised
               Learning by Measuring Eigenspectrum Decay},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {35},
  pages     = {17626--17638},
  year      = {2022},
  doi       = {10.52202/068431-1281},
  annote    = {Power-law exponent alpha per layer; VGG read at MaxPool/dropout
               inputs, ResNet at residual-block inputs. CNN intermediate alpha<1,
               deepest layers alpha~1 regardless of depth or SSL objective},
}

@inproceedings{garrido2023rankme,
  author    = {Garrido, Quentin and Balestriero, Randall and Najman, Laurent and
               LeCun, Yann},
  title     = {{RankMe}: Assessing the Downstream Performance of Pretrained
               Self-Supervised Representations by Their Rank},
  booktitle = {International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {10929--10974},
  year      = {2023},
  annote    = {arXiv:2210.02885. Entropy effective rank of embeddings from 25,600
               images, single layer; clipped at the embedding dimension 2048},
}
% UNVERIFIED: no DOI (PMLR); pages from the PMLR listing.

@inproceedings{thilak2024lidar,
  author    = {Thilak, Vimal and Huang, Chen and Saremi, Omid and Dinh, Laurent and
               Goh, Hanlin and Nakkiran, Preetum and Susskind, Joshua M. and
               Littwin, Etai},
  title     = {{LiDAR}: Sensing Linear Probing Performance in Joint Embedding
               {SSL} Architectures},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  annote    = {arXiv:2312.04000. Effective rank of the LDA matrix; single layer},
}
% UNVERIFIED: no DOI (ICLR).

@inproceedings{zhuo2023rankDifferential,
  author    = {Zhuo, Zhijian and Wang, Yifei and Ma, Jinwen and Wang, Yisen},
  title     = {Towards a Unified Theoretical Understanding of Non-contrastive
               Learning via Rank Differential Mechanism},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  annote    = {arXiv:2303.02387. Effective rank of online vs target outputs over
               epochs},
}
% UNVERIFIED: no DOI (ICLR).

@inproceedings{jing2022dimensionalCollapse,
  author    = {Jing, Li and Vincent, Pascal and LeCun, Yann and Tian, Yuandong},
  title     = {Understanding Dimensional Collapse in Contrastive Self-Supervised
               Learning},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  annote    = {arXiv:2110.09348. Embedding covariance spectrum only},
}
% UNVERIFIED: no DOI (ICLR).

@article{smirnov2026erank,
  author  = {Smirnov, Maksim and Kononov, Grigory and Linich, Anastasiia and
             Surkov, Egor and Shvetsov, Egor},
  title   = {{ERank} in Latent Space as an Image-Complexity and Richness Measure},
  journal = {arXiv preprint arXiv:2607.19315},
  year    = {2026},
  annote  = {Effective rank of the channel covariance of ONE image's HW x C
             feature map (positions as samples), ResNet-18 layers 1-4 and CLIP;
             the only other work using spatial positions as samples},
}

@inproceedings{chun2026estimating,
  author    = {Chun, Chanwoo and Canatar, Abdulkadir and Chung, SueYeon and
               Lee, Daniel D.},
  title     = {Estimating Dimensionality of Neural Representations from Finite
               Samples},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  annote    = {arXiv:2509.26560. Participation ratio is strongly biased at small
               sample sizes; bias-corrected estimator; LLM layer profiles},
}
% UNVERIFIED: no DOI (ICLR).

@article{zhou2026intrain,
  author  = {Zhou, Qinqin and Chen, Fuhai and Wu, Jipeng and Chen, Zhiwei and
             Hu, Zhikai and Cai, Weiwei},
  title   = {{InTrain}: Intrinsic Trainability for Zero-Cost Neural Architecture
             Search},
  journal = {arXiv preprint arXiv:2606.18676},
  year    = {2026},
  annote  = {Participation ratio of per-layer activation covariance in untrained
             networks as a NAS proxy},
}

% ===== Rank along depth: theory, gradients, transformers =================

@article{baker2024gradientRank,
  author  = {Baker, Bradley T. and Pearlmutter, Barak A. and Miller, Robyn and
             Calhoun, Vince D. and Plis, Sergey M.},
  title   = {Low-Rank Learning by Design: The Role of Network Architecture and
             Activation Linearity in Gradient Rank Collapse},
  journal = {arXiv preprint arXiv:2402.06751},
  year    = {2024},
  annote  = {Gradient rank <= min(rank Z_{i-1}, rank Delta_i); explicit
             stride/kernel bounds for convolutions; numerical rank of gradients,
             activations and deltas per layer over training},
}

@article{makwana2025oscillations,
  author  = {Makwana, Darshan},
  title   = {Some Theoretical Results on Layerwise Effective Dimension
             Oscillations in Finite Width {ReLU} Networks},
  journal = {arXiv preprint arXiv:2507.07675},
  year    = {2025},
  annote  = {Expected rank of the m x n hidden activation matrix of random ReLU
             layers: geometric decay with ratio 1-2/pi and revival depths},
}

@article{everett2026transformingRank,
  author  = {Everett, Katie},
  title   = {Transforming Rank: How Architecture Navigates the Spectral
             Pathologies of Depth},
  journal = {arXiv preprint arXiv:2607.14018},
  year    = {2026},
  annote  = {Effective rank of the input-output Jacobian at initialisation,
             normalised by width d; skip connections restore rank; 4x width
             expansion keeps the branch Jacobian full rank (Marchenko-Pastur)},
}

@article{jha2025spectralScaling,
  author  = {Jha, Nandan Kumar and Reagen, Brandon},
  title   = {Spectral Scaling Laws in Language Models: How Effectively Do
             Feed-Forward Networks Use Their Latent Space?},
  journal = {arXiv preprint arXiv:2510.00537},
  year    = {2025},
  annote  = {Participation ratio and Shannon rank of FFN post-activation
             covariance per layer, normalised by hidden width D ("utilisation");
             sub-linear growth of rank with width},
}

@inproceedings{skean2025layerByLayer,
  author    = {Skean, Oscar and Arefin, Md Rifat and Zhao, Dan and Patel, Niket
               and Naghiyev, Jalal and LeCun, Yann and Shwartz-Ziv, Ravid},
  title     = {Layer by Layer: Uncovering Hidden Representations in Language
               Models},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
  annote    = {arXiv:2502.02013. Matrix-based entropy per layer for LLMs and
               vision transformers; mid-depth compression},
}
% UNVERIFIED: venue inferred from the arXiv v2 header; no DOI found.

% ===== Pruning with activation rank =====================================

@inproceedings{sui2021chip,
  author    = {Sui, Yang and Yin, Miao and Xie, Yi and Phan, Huy and
               Aliari Zonouz, Saman and Yuan, Bo},
  title     = {{CHIP}: {CHannel} Independence-based Pruning for Compact Neural
               Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  year      = {2021},
  annote    = {arXiv:2110.13981. Nuclear-norm change of the matricised
               feature-map set when one map is removed; HRank successor},
}
% UNVERIFIED: no DOI found via Crossref.

% ===== Adjacent (separability, not dimensionality) =======================

@article{he2023separation,
  author  = {He, Hangfeng and Su, Weijie J.},
  title   = {A Law of Data Separation in Deep Learning},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {120},
  number  = {36},
  pages   = {e2221704120},
  year    = {2023},
  doi     = {10.1073/pnas.2221704120},
  annote  = {Separation fuzziness improves at a constant geometric rate per layer;
             not a dimensionality measure},
}
% UNVERIFIED: issue/article number.

@inproceedings{sukenik2024ncLowRank,
  author    = {S{\'u}ken{\'i}k, Peter and Mondelli, Marco and Lampert, Christoph},
  title     = {Neural Collapse versus Low-Rank Bias: Is Deep Neural Collapse
               Really Optimal?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  doi       = {10.52202/079017-4388},
  annote    = {arXiv:2405.14468. Singular values of MLP-head layers on a ResNet20
               backbone; low-rank solutions instead of DNC in deep/wide settings},
}

@article{galanti2022minimalDepth,
  author  = {Galanti, Tomer and Galanti, Liane},
  title   = {On the Implicit Bias Towards Minimal Depth of Deep Neural Networks},
  journal = {arXiv preprint arXiv:2202.09028},
  year    = {2022},
  annote  = {Effective depth via NCC separability per layer; not a rank measure},
}

@inproceedings{benshaul2022ncc,
  author    = {Ben-Shaul, Ido and Dekel, Shai},
  title     = {Nearest Class-Center Simplification through Intermediate Layers},
  booktitle = {Topological, Algebraic and Geometric Learning Workshops (TAG-ML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {196},
  pages     = {37--47},
  year      = {2022},
  annote    = {arXiv:2201.08924. NCC mismatch per layer},
}
% UNVERIFIED: PMLR volume number.

@inproceedings{doimo2020nucleation,
  author    = {Doimo, Diego and Glielmo, Aldo and Ansuini, Alessio and
               Laio, Alessandro},
  title     = {Hierarchical Nucleation in Deep Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  year      = {2020},
  annote    = {arXiv:2007.03506. Density peaks / neighbourhood overlap per
               ResNet block, 90,000 ImageNet images; not a dimension measure},
}
% UNVERIFIED: no DOI found via Crossref.

@article{zhang2026residual,
  author  = {Zhang, Xiao and Jiang, Ruoxi and Gao, William and Willett, Rebecca
             and Maire, Michael},
  title   = {Residual Connections Harm Generative Representation Learning},
  journal = {arXiv preprint arXiv:2404.10947},
  year    = {2026},
  annote  = {Decaying identity shortcuts lowers feature effective rank and
             improves probing in MAE/ViT; shortcuts "inject an echo of
             shallower representations"},
}
% UNVERIFIED: v5 May 2026; venue unknown.

@article{wen2025cbn,
  author  = {Wen, Yuxiao and Jacot, Arthur},
  title   = {Which Frequencies Do {CNNs} Need? Emergent Bottleneck Structure in
             Feature Learning},
  journal = {arXiv preprint arXiv:2402.08010},
  year    = {2025},
  annote  = {Convolution Bottleneck rank (theory); parameter norm scales as
             depth times CBN rank},
}
% UNVERIFIED: likely ICML 2024; not checked.
```

---

## 6. 搜尋的侷限與沒開到的東西

- **工具**：只用 WebSearch（美國索引）、arXiv PDF 全文、Crossref API。沒有查
  Google Scholar 的「被引用」清單，也沒有用 Semantic Scholar。Masarczyk 2023 與
  Ansuini 2019 的引用數都是三位數，靠關鍵字搜必有漏網；建議投稿前用 Scholar 對
  這兩篇各做一次「cited by」＋ `rank OR dimension` 篩選。
- **只讀摘要或部分**：Kong 2022（只 grep，未讀 α 逐層數值）；Skean 2025 視覺部分
  只掃 §6 標題；Doimo 2025 博士論文（arXiv 2510.21582，ResNet152 ID scale
  analysis）只看目錄；Súkeník et al. 2023（DUFM，純理論）未開；ToaSt（arXiv
  2602.15720）與「Estimating the Effective Rank of ViTs」（arXiv 2512.00792）皆 ViT
  專屬，未開；Chen & Bonner 的「無 ED 剖面」是 grep 未命中，不是通讀確認。
- **Nature 全文打不開**（302 到登入頁），Cohen 2020 用 bioRxiv v1 全文代替；正式版
  圖號與 SI 編號可能不同，引用時以 Nat. Commun. 版為準再核。
- **搜尋引擎誤導一例**：搜到「Dynamics of Feature Rank in ResNets（arXiv
  2404.10947）」，實際上該編號是 Zhang et al.「Residual Connections Harm Generative
  Representation Learning」（MAE／ViT），與 CNN 逐層秩無關，已按實際內容歸類。
- **DOI**：NeurIPS 2019／2020（Ansuini、Doimo、Daneshmand）在 Crossref 查不到
  Curran DOI；PMLR／ICLR／TMLR 本無 DOI。Kubaty 2025、Zhang 2026、Wen & Jacot
  的正式出處未能確認。
- **2024–2026 的 CNN 通道秩量測**：兩輪針對 ConvNeXt／EfficientNet／MobileNet 的
  搜尋都沒找到逐層激活秩剖面；最接近的是 Smirnov 2026（單張影像、ResNet-18
  四層）與 Zhou 2026（未訓練 NAS 候選）。不能排除 pruning 期刊（Neural Networks、
  TNNLS）裡有類似量測而未被索引到；`rmax-novelty` §5 的同一警告仍有效。
- **未核對的細節**：Ansuini PC-ID 的變異門檻百分比；Masarczyk 數值秩用的影像張數；
  Rangamani 的初始化對照是否涵蓋穩定秩；Konz 是否用 ImageNet 預訓練起點。

---

## 附錄 A：檢查過但排除於表格的相鄰工作

| 工作 | 為何排除 | 與我們的關係 |
|---|---|---|
| Doimo et al., NeurIPS 2020 | 量鄰域重疊與密度峰，不量維度 | 引 Ansuini 的 ID 說 conv3 ID 最高；可作旁證 |
| Galanti & Galanti 2022；Ben-Shaul & Dekel 2022；He & Su, PNAS 2023 | 逐層可分性／NCC／separation fuzziness | 「中間層神經崩塌」的可分性版，與秩無關 |
| Súkeník et al. 2023／2024；Zangrando et al. 2024 | 理論（DUFM）＋權重奇異值 | 「低秩偏差 vs DNC」，對象是權重與 MLP head |
| Wen & Jacot 2025 | CBN rank 理論 | 卷積瓶頸秩的理論定義，可在 r_max 段提一句 |
| Ziyin et al., ICLR 2025（CRH） | 表徵／權重／梯度對齊，不是維度剖面 | 無 |
| Sciandra et al. 2026 | Inverse Fisher Criterion 逐層逐 epoch | 「extractor 早期定型」的又一旁證，非秩 |
| Zhang et al. 2026（residual harm） | MAE ViT，特徵 erank 對 epoch | 捷徑抬高秩的方向性旁證（§4.3） |
| Nanda et al., NeurIPS 2023（已引） | 隨機神經元子集的冗餘，主要末層 | 無逐層剖面 |
