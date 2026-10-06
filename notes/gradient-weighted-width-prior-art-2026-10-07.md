# 梯度加權的寬度、任務相關維度、層剪枝：先行工作查核（2026-10-07）

緣起：作者 10-06 三問（正交化換到什麼、PCA／效率方向能不能加、能不能減層數），總帳「還沒查」表的
「梯度加權的寬度」列。範圍：Fisher／梯度加權的截尾與 PCA、active subspaces、task-relevant dimension、
剪枝側的梯度判準、層剪枝與 tunnel。方法：WebSearch 八個查詢 → 13 篇 PDF 下載到 `refs/B11/`（不進 git）→
`pdftotext` 讀原文的定義段、定理段、結果段；Crossref／proceedings 核對書目。每篇兩欄：做了什麼、留下什麼空格。

## 0. 結論

1. **「梯度加權的寬度」已有定義與量測：Cui, Zhang, Daulbaev, Gusak, Oseledets & Zhang, SIAM J. Math. Data Sci. 2020**。
   他們對第 l 層的中間向量 x（攤平：通道 × 位置）取損失梯度的非置中共變異數 C ＝ E[∇c(x)∇c(x)ᵀ]，
   定義「active neurons 數」n_{l,AS} ＝ 使前 i 個特徵值的累積比例 ≥ 1−ε 的最小 i（Def. 3.1）——**就是我們的 τ 尺，
   換成梯度 Gram**。VGG-19 CIFAR-10 第 4–7 層（8,192–16,384 個攤平神經元）的 active neurons 為 105／84／54／36。
   有 Poincaré 型誤差界（Lemma 2.3、3.1：損失近似誤差 ≤ O(λ_{r+1}+…+λ_n)）。**我們 B7 量的 k*_τ(Γ_end)（通道層級）是同一個量的
   通道版本，論文裡必須寫成「Cui 等人的 active-neuron count 在通道共變異數上的版本」，不能寫成新量。**
2. **Γ 本身是 K-FAC 的 S 因子**（Grosse & Martens, ICML 2016；Martens & Grosse 2015）：卷積層 Fisher ≈ S ⊗ A，
   S ＝ E[∇_{s_i}L ∇_{s_i}Lᵀ] 對位置 i 平均（非置中、位置解析、c_out × c_out），A ＝ patch 共變異數（c_in k² × c_in k²）。
   **A 就是 Σ_patch、S 就是 Γ**——EigenDamage（Wang, Grosse, Fidler & Zhang, ICML 2019）在 Q_S ⊗ Q_A 的特徵基底裡做通道剪枝。
   論文談 Σ_patch 與 Γ 時要引這兩篇定位：這兩個矩陣在二階最佳化文獻裡有名字。
3. **「變異大 ≠ 任務需要」有直接的先行證據：Hsu et al., ICLR 2022（FWSVD）**。BERT STS-B 把每層奇異值分十組逐組截掉：
   最小的第 10 組造成的掉分與第 2 組一樣大（Fig. 1）。他們的修法是以列和的經驗 Fisher 當權重做加權 SVD（式 5–6）。
   後續：GFWSVD（Chekalina et al. 2025–26，Kronecker 兩因子加權）、AIR（Harder et al. 2026，activation-aware 起點＋影響矩陣）。
   都是**權重**的截尾、都在 transformer／LLM；沒有一篇把「達 τ 變異的主成分數」與「達 τ 梯度能量的主成分數」逐層並列。
4. **剪枝側的梯度判準是逐通道的重要度，不是寬度**：Taylor-FO（Molchanov et al., CVPR 2019：gate 上 (∂E/∂z_m)²＝(Σ_s g_s w_s)²）、
   Fisher pruning（Theis et al. 2018）。它們在標準基底、不取特徵分解，所以不產生「這一層要幾個方向」的數。
5. **層剪枝（作者第四問「減層數」）**：Chen & Zhao, **TPAMI 41(12):3048–3056, 2019**——逐層線性探針，相鄰層探針準確率增益 < 1.5% 的層
   整層拿掉再蒸餾重訓；VGG-16 在 CIFAR 三個資料集**末幾層都飽和**，CIFAR-100 砍掉 87% 參數不掉分；ResNet-56 在 CIFAR-10／SVHN
   **近半數卷積層**貢獻不足，砍 42.3% 參數準確率反升。Zhang, Bengio & Singer, **JMLR 23(67):1–28, 2022**——訓練後把單層重設為初始值：
   VGG 底層 critical、頂層 robust；ResNet 的 residual block 連重新隨機化都 robust；**任務越難 critical 層越多**（ResNet-18 在 ImageNet 上
   原本 robust 的 block 變敏感）。Pons et al. 2024（CKA 判準，ResNet-56 CIFAR-10 砍 >75% FLOPs 不掉分）。
   **與我們的關係**：可拿掉的層（VGG 末段、CIFAR）正是尺讀出閒置寬度（VGG-16 CIFAR-10 末段 0.30）與 B9 逐位置塌縮的層；
   任務變難兩者都消失（Zhang 的 ImageNet 對我們的 A26／ImageNet 摺入）。**沒有人把「層可拿掉」與「層的寬度塌縮」放在同一個網路上對照**；
   我們也沒量過層的可移除性——Limitations 要寫「我們量一層多寬，不量一層要不要」，並引這兩篇。
6. **總結**：梯度加權寬度的**定義**（Cui 2020）、**矩陣**（K-FAC）、**變異≠任務**的證據（FWSVD）都已知；仍是我們的是
   (a) Σ_out 與 Γ 的 τ 寬度在**同一層、同一批影像、沿訓練與 λ** 的並列（B7：P-b7.2 順序 +0.86；P-b7.5 數目高估 1.6–2.1 倍；λ 掃描
   末三層 Σ_out 寬度 32–39 對 Γ_end 179–312——**梯度 Gram 比輸出共變異數寬一個量級**，不是更窄）；(b) 這個並列對類別數與步數的
   反應（P-b7.3、P-b7.6）。**作者 10-06 的直覺「用梯度加權可能讀出比變異更窄、更接近任務需要的寬度」在我們 CIFAR-10 的資料上方向相反：
   塌縮的是變異，梯度仍張滿 ≥ 3K 個方向（P-b7.2 的 63/63）。** ImageNet 上沒有 Γ 的量測，不能外推。

## 1. 逐篇

### 1.1 Cui, Zhang, Daulbaev, Gusak, Oseledets & Zhang, "Active Subspace of Neural Networks: Structural Analysis and Universal Attacks", SIAM J. Math. Data Sci. 2(4):1096–1122, 2020（讀 §2、§3.2、§5.1；`cui2020_active_subspace.pdf`）

| 做了什麼 | 留下什麼空格 |
|---|---|
| Active subspace（Constantine）搬到網路中間層：C ＝ E[∇_x c_l(x) ∇_x c_l(x)ᵀ]，x 是第 l 層輸出攤平（通道 × 位置），c_l ＝ 後半網路的交叉熵。Def. 3.1 的 n_{l,AS}＝累積特徵值比例 ≥ 1−ε 的最小 i。Lemma 2.3（Poincaré）：E[(c−g(z))²] ≤ O(λ_{r+1}+…+λ_n)；Lemma 3.1 把它推到 conv／ReLU／BN／max-pool／CE 網路。Remark 2.1 明寫與 PCA 的差別：PCA 用資料共變異數，AS 用梯度共變異數，資料方向的變異不保證改變函數值。 | 只在**截斷點**一層做（cut-off layer l，之後接 PCE／邏輯回歸），不是逐層剖面；x 是攤平向量，n_{l,AS} 混了位置與通道；只有 VGG-19／ResNet-110 CIFAR-10／100；沒有隨機對照、沒有沿訓練、沒有與資料共變異數的 k* 並列（Fig. 5b 只比準確率：AS+PCE 勝 PCA+LR）。 |
| 數字：VGG-19 CIFAR-10 第 4–7 層 active neurons 105／84／54／36（Fig. 5a 奇異值近指數衰減）；r＝50 時 ASNet 14.43× 儲存、3.44× 計算，ASNet-s 23.98×／7.30×。 | 我們的 k*_τ(Γ) 是通道層級、逐層、21 個 checkpoint＋λ 掃描、與 k*_τ(Σ_out) 並列——這是空格。但**定義與誤差界要歸他們**。 |

### 1.2 Meneghetti, Demo & Rozza, "A dimensionality reduction approach for convolutional neural networks", Applied Intelligence 53(19):22818–22833, 2023（讀 §1–2、§4；`meneghetti2023_as_cnn.pdf`）

| 做了什麼 | 留下什麼空格 |
|---|---|
| Cui 的框架一般化：降維用 AS 或 POD（＝資料 PCA），映射用 PCE 或 FNN；VGG-16 CIFAR-10（77.98%）與自訂資料集；r 固定 50（沿用 Cui）。POD+FNN 不需要額外梯度、訓練更快，準確率與 AS 相當或更好。 | 沒有新的寬度量；r 不是從譜讀出來的。對我們只有一句：在他們的設定裡**資料 PCA 與梯度 AS 的降維效果相當**（表 2），與 Cui Fig. 5b 的方向不完全一致。 |

### 1.3 Grosse & Martens, "A Kronecker-factored approximate Fisher matrix for convolution layers", ICML 2016, PMLR 48:573–582；Wang, Grosse, Fidler & Zhang, "EigenDamage", ICML 2019, PMLR 97:6566–6575（讀 EigenDamage §2、§4；`wang2019_eigendamage.pdf`）

| 做了什麼 | 留下什麼空格 |
|---|---|
| 卷積層 Fisher ≈ S ⊗ A，A ＝ (1/|I|) Σ_i E[a_i a_iᵀ]（patch，c_in k² 維）、S ＝ E[∇_{s_i}L ∇_{s_i}Lᵀ]（輸出端每個位置的梯度，c_out 維；假設空間均勻、位置間梯度不相關）。EigenDamage 把權重旋到 Q_S ⊗ Q_A 的基底，在該基底裡 Fisher 近對角（Fig. 2），逐列／逐行用 Θ ＝ W′ ⊙ diag(Λ_A)diag(Λ_S)ᵀ ⊙ W′ 當重要度剪通道；VGG-19／ResNet-32 CIFAR、Tiny-ImageNet；10× 參數、8× FLOPs。 | 他們的 A、S 正是我們的 Σ_patch、Γ_end（位置解析、非置中）。他們不報 A 或 S 的譜寬度、不談一層需要幾個方向、不看訓練過程、沒有隨機對照；全域比例決定逐層秩。**措辭**：Σ_patch ⊗ Γ 是 K-FAC 的 Fisher 近似，我們量的是這兩個因子各自的 τ 寬度與它們之間的轉移（B10）；B10 的 M̃ ＝ Γ^{−1/2}Σ_outΓ^{−1/2} 與 K-FAC 的 Kronecker 假設（S 與 A 獨立）是不同的問題，但要並提。 |

### 1.4 Hsu, Hua, Chang, Lou, Shen & Jin, "Language model compression with weighted low-rank factorization", ICLR 2022（讀 §2–4；`hsu2022_fwsvd.pdf`）；Chekalina et al., GFWSVD／MFF, arXiv 2505.17974（v2 2026-07）；Harder, Becking, Mueller & Samek, AIR, arXiv 2606.19993（讀摘要與圖 1）；Xu 2026 arXiv 2604.18085（讀摘要）

| 做了什麼 | 留下什麼空格 |
|---|---|
| FWSVD：BERT STS-B 奇異值分十組逐組截掉，**最小的第 10 組掉分與第 2 組相當**（Fig. 1）——「奇異值大小不反映任務重要度」。修法：經驗 Fisher Î_{W_ij}＝Σ_i (∂L/∂W_ij)²，列和 Î_{W_i}，解 min‖Î^{1/2}W − Î^{1/2}AB‖（式 6）＝對 Î^{1/2}W 做 SVD 再除回去。GLUE／CoNLL：FWSVD＋微調勝 SVD＋微調、追平 DistilBERT 而不需通用預訓練。GFWSVD：Fisher ≈ A ⊗ B（梯度側 × 激活側），Cholesky 兩邊加權後 SVD（閉式）；LLaMA-2-7B、Llama 3 不崩。AIR：從 SVD-LLM 的 activation-aware 最佳解出發，再用反向「影響矩陣」做一輪 ALS。Xu 2026：壓縮比 × stable rank 預測掉分（r 0.89／0.84）。 | 全是**權重矩陣**的低秩，對象是 transformer 的線性層；沒有卷積、沒有激活共變異數的寬度、沒有逐層剖面。對我們：(a) 「變異大 ≠ 任務需要」有先行證據，正文談摺入／尺時要引 FWSVD；(b) 我們的摺入（PCA 截尾＝activation-aware 的 Σ_in 加權）對應 ASVD／SVD-LLM 那一側，**不是**梯度加權那一側——作者問的「PCA 方向」文獻已經走到兩者合併（AIR），我們沒有優勢；(c) Xu 的「stable rank 預測掉分」是權重譜，我們的 §(ii) 摺入掉分對 τ 的關係是激活譜，可並提不重做。 |

### 1.5 Molchanov, Mallya, Tyree, Frosio & Kautz, "Importance Estimation for Neural Network Pruning", CVPR 2019, pp. 11256–11264（讀 §3）；Theis, Korshunova, Tejani & Huszár, "Faster gaze prediction with dense networks and Fisher pruning", arXiv 1801.05787（讀 §3.1）

| 做了什麼 | 留下什麼空格 |
|---|---|
| 一階 Taylor 重要度 I_m ＝ (g_m w_m)²；在 gate 上 I_m(z) ＝ (∂E/∂z_m)² ＝ (Σ_s g_s w_s)²（式 9）＝ 前一層參數的 group contribution；BN 後放 gate；全網一個排序、不需逐層敏感度；ImageNet ResNet-50／101。Fisher pruning 是同一個量的 Fisher 寫法，加算力項。 | 逐通道、標準基底——不是「這一層需要幾個方向」。與 Γ 的關係：gate 的 Fisher 就是 Γ 的對角線（各通道梯度能量，對位置求和後平方）；我們的 k*_τ(Γ) 用的是整個 Γ 的譜。論文一句話劃界即可（HRank／ThiNet／Taylor／Fisher 都是通道重要度；寬度是譜的計數）。 |

### 1.6 Xu & Yu, "Neural Networks Perform Sufficient Dimension Reduction", AAAI 2025（讀摘要、定理陳述）；Gulati, Abdelaleem, Sederberg & Nemenman, "Mutual Information and Task-Relevant Latent Dimensionality", PMLR 326:262–293, 2026（讀摘要）

| 做了什麼 | 留下什麼空格 |
|---|---|
| Xu & Yu：有秩正則時**第一層**權重張成 central mean subspace（定理 1 母體、定理 2 一致性）；回歸、全連接。Gulati：task-relevant dimension ＝ 保持互資訊所需的瓶頸維度，用 hybrid critic 估，單一過參數化模型一次讀出。 | 都是**全網一個數**（第一層或瓶頸），不是逐層；SDR 的 central subspace 由梯度外積估（Härdle–Stoker、Xia 的 OPG／MAVE）就是 AGOP 的統計學祖先——A32 的劃界句可加一筆「AGOP 的 SDR 來源」。Gulati 的「標準估計器高估維度」對我們無直接對應（我們不估互資訊）。 |

### 1.7 層剪枝：Chen & Zhao, TPAMI 41(12):3048–3056, 2019（讀 §3.1、§4.2）；Zhang, Bengio & Singer, JMLR 23(67):1–28, 2022（讀 §3.2）；Pons, Yamamoto, Reali Costa & Jordao, arXiv 2405.17081 v2（讀摘要）；Masarczyk 2023 tunnel（`s3-prior-art` §1.1，不重複）

| 做了什麼 | 留下什麼空格 |
|---|---|
| Chen & Zhao：每層接一個線性分類器，相鄰層增益 < 原模型 1.5% 的層判為冗餘，整層拿掉後以 KD 重訓；**VGG-16 在 CIFAR-10／100／SVHN 末幾層全部飽和**；ResNet-56 在 CIFAR-10／SVHN 近半層冗餘；VGG-16 CIFAR-100 砍 87% 參數不掉分、ResNet-56 CIFAR-10 砍 42.3% 參數反升；ResNet-101 MSCOCO 多標籤同法。 | 判準是探針準確率，**沒量層的寬度／秩**；沒有隨機對照、沒有 ImageNet 分類（ImageNet 只提 AlexNet 的 FLOPs）。「VGG 末段在 CIFAR 可拿掉」與我們「VGG 末段在 CIFAR 尺讀 0.30、逐位置塌縮」是同一群層的兩種讀法，沒人對照過。 |
| Zhang–Bengio–Singer：訓練後把一層重設為初始值（re-init）或重新抽（re-rand）；FCN／VGG 底層 critical、頂層 robust（robust 層訓練中幾乎不動）；ResNet 的 residual block 多半 robust 且連 re-rand 都 robust（skip 的響應大於殘差分支）；**任務越難 critical 層越多**（ResNet-18 的 block 在 ImageNet 變敏感）；ViT 附錄同趨勢。 | 量的是「這層的訓練有沒有用」，不是寬度；沒有譜。給我們的句子：閒置寬度與 robust 層都是「容量 > 任務」的表現，都隨任務變難消失——Limitations 引它說明我們不量深度軸。 |
| Pons et al. 2024：CKA（整層表徵對原網路的相似度）當層重要度，ResNet-56／110、ResNet-50；ResNet-56 CIFAR-10 砍 >75% FLOPs 不掉分、對抗與 OOD 更穩。 | 同上；CKA 已在我們 A20 用過（不同用途）。 |

## 2. 對作者 10-06 三問的更新

- **正交化換到什麼**：文獻側沒有新東西；D5 的定位不變（探針）。
- **PCA／效率方向能不能加**：文獻已走到 activation-aware（ASVD／SVD-LLM）＋ gradient-aware（FWSVD／GFWSVD／AIR）合併，對象是 LLM 權重；
  卷積側 Cui 2020 已把梯度 AS 做成壓縮方法。**加不進去**，維持 09-07／09-11 的定位。
- **梯度加權的寬度**：定義已知（Cui Def. 3.1）；我們已量（B7 的 k*_τ(Γ)），且方向是「Γ 比 Σ_out 寬」。能寫的是**並列**與**對 K、步數、λ 的反應**，
  寫法「Cui 等人的 active-neuron count 在通道共變異數上的版本，與變異計數逐層並列」。不需要新實驗。
- **減層數**：沒量過；文獻（Chen & Zhao、Zhang、Pons、Masarczyk）的可拿掉層與我們的閒置層在 VGG／CIFAR 上重合，但無人對照。
  **候選陳述（只寫、不跑、未登記）**：「VGG-16 CIFAR-10 全寬臂裡，相鄰層線性探針增益 < 1.5%（Chen & Zhao 門檻）的層，與 k*₀.₉₉₉/r_max 最低的層是同一組（末段三層）」。
  成本：Mac CPU，對既有 `results/a18_pt/` 的 checkpoint 跑逐層探針，約 1 小時；要不要做由作者定，依雷達它只動第 4 軸（實驗），不動前三軸。

## 3. 書目（已核）

| key | 來源核對 |
|---|---|
| `cui2020activeSubspace` | Crossref DOI 10.1137/19M1296070，SIMODS 2(4):1096–1122 |
| `grosse2016kfacConv` | proceedings.mlr.press v48，pp. 573–582 |
| `wang2019eigendamage` | proceedings.mlr.press v97，pp. 6566–6575 |
| `hsu2022fwsvd` | ICLR 2022（無 DOI；iclr.cc poster 6157、arXiv 2207.00112） |
| `molchanov2019importance` | Crossref DOI 10.1109/CVPR.2019.01152，pp. 11256–11264 |
| `chen2019shallowing` | Crossref DOI 10.1109/TPAMI.2018.2874634，TPAMI 41(12):3048–3056 |
| `zhang2022allLayers` | jmlr.org v23/20-069，JMLR 23(67):1–28 |

未進 bib（只在筆記）：Meneghetti 2023（Crossref 10.1007/s10489-023-04730-1）、Theis 2018（arXiv）、Chekalina 2025／26、Harder 2026、Xu 2026、Xu & Yu 2025、Gulati 2026、Pons 2024。
PDF 在 `refs/B11/`（不進 git）。
