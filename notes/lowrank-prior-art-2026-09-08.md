# 投影實驗的先行文獻查證（2026-09-08）

對應 `code/scripts/projection_check.py` 與 `results/projection/*.csv`。問題：「把每一個
conv 輸出在推論時投影到它自己的前 k\*(τ) 個主方向（全部層同時、不重訓、不微調），
τ = 0.999 時 ResNet-50 V1／ResNet-18／VGG-16 top-1 不變、ResNet-50 A1 掉 1.8 點；
τ = 0.95 時掉約 20 點」——這件事文獻裡有沒有人做過？該怎麼引用才不會宣稱已知的東西？

**結論：這個操作本身就是 Zhang, Zou, He, Sun（CVPR 2015／TPAMI 2016）的
「linear response reconstruction」（他們的 Eq. (4)），一字不差：對 conv 響應
（activation 之前）做 PCA、取前 d′ 個特徵向量、M = U_{d′}U_{d′}ᵀ、然後把 M 吸進
W。他們也用 PCA 累積能量當秩選擇準則、也對整個模型（VGG-16 全 13 層）做、也報告
不微調的結果。所以「投影 = 已知的 low-rank 壓縮」是對的，論文裡必須寫成
「Zhang et al. 的 linear 版本，拿我們量到的 k\*(τ) 當秩」，不得寫成新方法。**
還剩下的、查到的範圍內沒人做過的，只有三件小事（§4）：(a) 用**同一個變異數
門檻**套到全部層、報告整體 top-1（Zhang 用的是 greedy 能量×複雜度配置，不是統一
門檻）；(b) 在 ResNet-50 上、對照 r\_max；(c) 同一架構不同訓練配方之間可移除
比例不同。但 (c) 目前只有 A1 vs V1 兩個點，而且有一個層級的假象要先排除（§4.3）。

---

## 1. 直接先行文獻（必引）

### Zhang, Zou, He, Sun — TPAMI 2016（arXiv 1505.06798 v2）

「Accelerating Very Deep Convolutional Networks for Classification and Detection」，
IEEE TPAMI 38(10):1943–1955。CVPR 2015 版本（Zhang, Zou, Ming, He, Sun,
「Efficient and Accurate Approximations of Nonlinear Convolutional Networks」）是
同一套方法的前身，差別見下。這是我們投影實驗的直接先例，讀了全文。

**(a) 他們做的分解就是我們的投影。** §3.1：

> "Our assumption is that the filter response at a pixel of a layer approximately
> lies on a low-rank subspace."
>
> "Under the assumption that the vector y is on a low-rank subspace, we can write
> y = M(y − ȳ) + ȳ, where M is a d-by-d matrix of a rank d′ < d and ȳ is the mean
> vector of responses."
>
> "This problem can be solved by SVD [31] or actually Principal Component Analysis
> (PCA): let Y be the d-by-n matrix concatenating n responses with the mean
> subtracted, compute the eigen-decomposition of the covariance matrix YYᵀ = USUᵀ
> where U is an orthogonal matrix and S is diagonal, and M = U_{d′}U_{d′}ᵀ where
> U_{d′} are the first d′ eigenvectors. With the matrix M computed, we can find
> P = Q = U_{d′}."

y = Wx 是 conv 的**線性**響應（activation 之前），與我們量的對象相同。
「mean + P_k(y − mean)」正是他們的 y = M(y − ȳ) + ȳ。M = PQᵀ 之後
W′ = QᵀW 就是使用者說的 V_k(V_kᵀW)。

**(b) 他們早就畫過 PCA 能量曲線，而且數字和我們同量級。** §3.1、Fig. 2：

> "For example, in the Conv2 layer (d = 256) the first 128 eigenvectors contribute
> over 99.9% energy; in the Conv7 layer (d = 512), the first 256 eigenvectors
> contribute over 95% energy. This indicates that we can use a fraction of the
> filters to precisely approximate the original filters."
>
> "The low-rank behavior of the responses y is because of the low-rank behaviors of
> the filter weights W and the inputs x."

Fig. 2 圖說："These figures are obtained from 3,000 randomly sampled training
images."（SPP-10，OverFeat-7 變體，不是 ResNet。）

**(c) 能量對準確率：他們有 Fig. 3，逐層獨立。** §3.4 與 Fig. 3 圖說：

> "We empirically observe that the PCA energy after approximations is roughly
> related to the classification accuracy. ... Fig. 3 shows that the classification
> accuracy is roughly linear on the PCA energy."
>
> "Each layer is evaluated independently, with other layers not approximated."

也就是說「τ 越低掉越多、τ = 0.999 幾乎不掉」在單層上他們 2015 年就畫出來了。
我們的曲線是全部層同時投影的版本。

**(d) 秩選擇規則：不是變異數門檻，是 greedy 的能量乘積×複雜度配置。** §3.4：

> "To simultaneously determine the rank for each layer, we further assume that the
> whole-model classification accuracy is roughly related to the product of the PCA
> energy of all layers."

目標 E = ∏_l Σ_{a=1}^{d′_l} σ_{l,a}，在 Σ_l (d′_l/d_l)·C_l ≤ C 之下最大化；解法：

> "We initialize d′_l as d_l, and consider the set {σ_{l,a}}. In each step we remove
> an eigenvalue σ_{l,d′_l} from this set, chosen from a certain layer l. ... The
> eigenvalue σ_{l,d′_l} that has the smallest value of this measure [(ΔE/E)/ΔC] is
> removed. ... This step is greedily iterated, until the constraint of the total
> complexity is achieved."

所以每層的秩由**速度預算**決定，不是由「保留 τ 的變異數」決定。他們的 Table 6
「no rank selection」對照組是「每層相同 speedup ratio」，也不是統一能量門檻。
**查到的範圍內，「每層各自取 k\*(τ)、同一個 τ」這種配置的整體準確率，Zhang
沒有報告。**

**(e) 非線性與 asymmetric 版本才是他們的主菜；linear 版本被他們自己比下去。**
§3.2 把目標改成 Σ‖r(y_i) − r(My_i + b)‖²（r = ReLU），§3.3 再改成
asymmetric（第二項用前面已近似層的輸入 x̂）。§4.1：

> "Fig. 4 shows that the nonlinear solution consistently performs better than the
> linear solution."

我們做的是他們的 linear、symmetric 版本，也就是他們的**基線**。

**(f) 微調：TPAMI 版有，但主結果不微調。** §3.6：

> "However, we empirically find that fine-tuning is very sensitive to the
> initialization (given by the approximated model) and the learning rate."
>
> "Fortunately, our method has achieved very good accuracy even without fine-tuning
> as we will show by experiments. ... In our experiments, we use a learning rate of
> 1e-5 and a mini-batch size of 128, and fine-tune the models for 5 epochs in the
> ImageNet training data. We note that in the following the results are without
> fine-tuning unless specified."

**(g) 不微調的整模型結果（VGG-16，13 層全做）。** §4.2 與 Table 7：

> "Somewhat surprisingly, our method has demonstrated compelling results for this
> very deep model, even without fine-tuning. Our no-fine-tuning model has a 0.9%
> increase of 1-view top-5 error for a speedup ratio of 4×. ... After fine-tuning,
> our model has a 0.3% increase of 1-view top-5 error for a 4× speedup."

Table 7（VGG-16 基線 1-view top-5 error 10.1%；表列**增加量**）：

| speedup | 3× | 4× | 5× |
|---|---|---|---|
| Jaderberg et al. [17]（他們重作） | 2.3 | 9.7 | 29.7 |
| our asym. (3d)，不微調 | 0.4 | 0.9 | 2.0 |
| our asym. (3d) + FT | 0.0 | 0.3 | 1.0 |

另外 Table 6（無 3d 分解）："For a 4× speedup, the rank selection reduces the
increased error from 6.38% to 3.84%." 以及對 Lebedev 等人的評語：

> "Note that results in [49] are after fine-tuning. This suggests that fine-tuning
> is not sufficient for whole-model acceleration; a good optimization solver for the
> decomposition is needed."

**(h) 樣本數。** CVPR 版 §2.5："These simple and close-form solutions can produce
good results using a very small subset of training images (3,000 out of one
million)." 我們用 3,200 張校準，同量級。

**CVPR 2015 與 TPAMI 2016 的差別**：CVPR 版摘要 "A whole-model speedup ratio of
4× is demonstrated on a large network trained for ImageNet, while the top-5 error
rate is only increased by 0.9%."——那是 SPP-10、無微調。TPAMI 版加了 VGG-16、
3d 分解、微調、偵測。Kim et al.（ICLR 2016）的描述可以直接引："they also present
a rank selection method based on PCA accumulated energy and an optimization method
which minimizes the reconstruction error of non-linear responses. In the extended
version (Zhang et al., 2015a), the additional fine-tuning of entire network was
considered for further improvement."

### Denton, Zaremba, Bruna, LeCun, Fergus — NeurIPS 2014（arXiv 1404.0736）

「Exploiting Linear Structure Within Convolutional Networks for Efficient
Evaluation」。權重側低秩（SVD／CP／biclustering），**沒有秩選擇規則**（K 是掃出來
的，Fig. 3–4 直接標 K 值），**有微調**，只做前兩層。

> 摘要："we demonstrate speedups of convolutional layers on both CPU and GPU by a
> factor of 2×, while keeping the accuracy within 1% of the original model."
>
> §3.5 Fine-tuning："Many of the approximation techniques presented here can
> efficiently compress the weights of a CNN with negligible degradation of
> classification performance provided the approximation is not too harsh.
> Alternatively, one can use a harsher approximation that gives greater speedup
> gains but hurts the performance of the network. In this case, the approximated
> layer and all those below it can be fixed and the upper layers can be fine-tuned
> until the original performance is restored."
>
> §4："Unless stated otherwise, classification numbers refer to those of fine-tuned
> models." "we restrict our attention to the first and second convolutional layers
> in our speedup experiments."

有一點和「資料相依」沾邊：Eq. (3) 的 data-covariance 範數
‖W‖_data = ‖Σ̂^{1/2} W_F‖_F，"this approach adapts to the input distribution
without the need to iterate through the data"。這是**輸入**共變異數加權，不是輸出
響應 PCA。

### Jaderberg, Vedaldi, Zisserman — BMVC 2014（arXiv 1405.3866）

「Speeding up Convolutional Neural Networks with Low Rank Expansions」。空間可分離
＋跨通道低秩；秩（M、K）是掃描；有 "filter reconstruction" 與 "data reconstruction"
兩種最佳化；後者可接微調；只做 Conv2、Conv3；主結果是場景文字字元分類，不是
ImageNet。

> 摘要："showing a possible 2.5× speedup with no loss in accuracy, and 4.5× speedup
> with less than 1% drop in accuracy"
>
> §2.2.2："one can optimize a scheme's separable basis by aiming to reconstruct the
> outputs of the original convolutional layer given training data."
>
> §3："For data reconstruction optimization, we optimize each approximated layer in
> turn, and can incorporate a fine-tuning with joint optimization." "For the CNN
> presented, we only approximate layers Conv2 and Conv3."
>
> ImageNet 只有一句："simply doing a filter reconstruction approximation with
> Scheme 2 of the second layer of OverFeat [32] gives a 2× theoretical speedup with
> only 0.5% drop in top-5 classification accuracy on ImageNet"

### Garg, Panda, Roy — IEEE Access 2020（arXiv 1812.06224；refs.bib 已有 `garg2019lowEffort`）

這篇已是我們的驗證關卡，但它和投影實驗的關係要講清楚：他們用**同一個 99.9%
變異數門檻**、**全部層同時**、對 activation 做 PCA，算出「significant
dimensions」——**然後重新初始化、重訓一次**，不是投影、不是不微調。

> "We then determine the optimized network's layer-wise width from the number of
> principal components required to explain 99.9% of the cumulative explained
> variance. We call these the 'significant dimensions' of each layer"
>
> "Once the requisite width and depth are identified, the user can create a new,
> randomly initialized network of the identified width and depth and train once to
> get the final, efficient model."
>
> 樣本量："collecting data over enough mini batches such that D/M is roughly larger
> than 100 provides enough samples"（與我們 n/C ≥ 50 的門檻同精神）

所以「τ = 0.999 全部層同時」這個配置的**門檻**來自 Garg，**操作**來自 Zhang；
兩者合起來就是我們做的事。

## 2. 後續的秩選擇與「免微調」文獻（相關工作段落用）

| 文獻 | 秩怎麼選 | 資料相依？ | 微調？ | 與我們的關係 |
|---|---|---|---|---|
| Xue, Li, Gong, Interspeech 2013 | 權重 SVD，保留固定比例奇異值 | 否 | 是 | 權重側「能量門檻」的源頭（語音 DNN；**未開全文**） |
| Kim et al., ICLR 2016（Tucker） | "rank selection with variational Bayesian matrix factorization" | 否（權重） | "fine-tuning to recover accumulated loss of accuracy" | 明確把 Zhang 的 PCA 能量秩選擇列為對照 |
| Kim & Kyung, arXiv 1806.10821 (2018) | 組合最佳化＋線性準確率預測 | 否 | 是 | 觀察到 "the test accuracy before fine-tuning (i.e. after low-rank decomposition) is almost linear to the recovered accuracy after several training epochs"——支持「不微調的準確率是有意義的讀數」 |
| Idelbayev & Carreira-Perpiñán, CVPR 2020 | 秩當離散變數與權重聯合最佳化（LC 演算法） | 否 | 是（訓練即方法） | 秩選擇的「學出來」路線；BALF 說他們也有 fine-tuning-free 數字，**未在原文確認** |
| Liebenwein et al., NeurIPS 2021（ALDS） | Eckart–Young–Mirsky 逐層誤差上界，"minimize the maximum compression error across layers" | 否（權重 SVD） | 主結果有；Fig. 5(a) 有 "Compress-only (r = 0)" | 基線清單裡就有 "PCA (Zhang et al., 2015b)" 與 "SVD with energy-based layer allocation (SVD-Energy)"——**能量門檻配置在 2021 年已是標準基線** |
| Papadimitriou & Jain, arXiv 2107.05787 (2021, DDLR) | 每個 FC 層在效能保證下取最低秩，凸鬆弛 | 是 | "without requiring any retraining" | 只做全連接層 |
| Chee, Renz, Damle, De Sa, NeurIPS 2022（MPC） | interpolative decomposition，ε 決定每層 k | 是（post-activation 輸出） | "eliminate the need for fine tuning" | 是通道**挑選**（結構化剪枝＋修正矩陣），不是旋轉到主方向；有 VGG-16 ImageNet 的 "Pre-fine tuning" 曲線（Fig. 2） |
| Zhang, Liu, Weng, J. Real-Time Image Process. 20(4):64, 2023 | "cosine similarity SVD" | ? | 標題即 "without fine-tuning" | **未能開啟全文**（Springer 與 ACM 都擋）；投稿前要看它有沒有 ResNet-50 數字 |
| Zhang & Saab, ACHA 82:101837 (2026; arXiv 2502.02766) | 理論 | 是 | — | "we develop an analytical framework for data-driven post-training low-rank compression. We prove three recovery theorems under progressively weaker assumptions about the approximate low-rank structure of activations"；引言："These data-driven methods tend to perform well in practice, even before fine-tuning" |
| González-Martínez, arXiv 2509.25136 v3（2026, BALF） | 每層 activation 能量 E_l(P_l) 的全域 knapsack，Lagrangian 鬆弛 | 是（**輸入**白化，SVD-LLM 路線） | "Fine-Tuning-Free" | 見下 |

**BALF 值得多寫幾句**，因為它是 2025–26 年 CNN 免微調低秩的現況，而且它自己把
能量門檻法定位成舊法：

> "Energy-based selection then consists of picking the lowest rank such that a
> certain energy threshold (manually defined by the user) is retained. We propose
> instead to solve an optimization problem that maximizes the total retained energy
> globally"
>
> 腳註 1："Zhang et al. (2016) explore a similar data-aware approach in the image
> classification domain, but they also take nonlinearities into account."
>
> 結果："we can reduce the FLOPs of the ResNeXt-101 model by 50% with only about
> 1.5 percentage points of top-1 accuracy drop."（ResNet-50 有曲線，Fig. 5(b)，
> 沒有單句可引。）

兩個和我們直接相關的技術點：
1. BALF 是**輸入側**（白化 X，再對 W 做 SVD），Zhang 與我們是**輸出側**（響應
   PCA）。兩者在 ReLU 之前的 conv 都是資料相依低秩，但不是同一個投影。
2. BALF 明講 "our whitening-based formulation is more general than prior ones,
   naturally covering rank-deficient activations"——1×1 擴張層輸入秩虧損正是我們
   r\_max 要處理的現象，他們從壓縮那邊撞到同一件事，但沒有把它寫成 min(C_in·k², C_out)
   的上界，也沒有拿來正規化。可以和 `rmax-novelty-2026-09-03.md` 並列。

## 3. 訓練配方對可移除比例的影響：文獻裡有沒有人講過？

**低秩／PCA 這條線：查到的範圍內沒有。** 沒有任何一篇比較同一架構在不同 ImageNet
配方（V1 vs A1、CE vs BCE、有無 mixup/cutmix）下的響應 PCA 可截斷比例。
最接近的四類：

1. **剪枝界的共識句**（無資料相依低秩）。Kuznedelev, Kurtić, Frantar, Alistarh,
   CAP, NeurIPS 2023，摘要："These highly-accurate models are challenging to deploy,
   as they appear harder to compress using standard techniques such as pruning."
   引言："modern training procedures involve longer training schedules together
   with a careful choice of hyperparameters (learning rate schedule, regularization,
   augmentation)". 講的是 ViT／ConvNeXt 對 ResNet-50、非結構化剪枝、**有微調**；
   不是同架構跨配方。
2. **損失函數只改最後幾層**。Kornblith, Chen, Lee, Norouzi, NeurIPS 2021：
   "Using centered kernel alignment to measure similarity between hidden
   representations of networks, we find that differences among loss functions are
   apparent only in the last few layers of the network." "Consistent differences
   among loss functions are present only in the last third of the network, starting
   around block 13." "All objectives that improve accuracy over softmax cross-entropy
   also lead to greater separation between representations of different classes in
   the penultimate layer features. These alternative objectives appear to collapse
   within-class variability". 他們只換損失（含 sigmoid/BCE、label smoothing），
   **不換增強與訓練長度**；A1 同時換了 BCE、mixup/cutmix、RandAugment、600 epoch、
   LAMB。所以 A1 的逐層 k\*/r\_max 全網偏高（0.895 vs 0.858，τ = 0.999）不與
   Kornblith 矛盾，但也不能拿 Kornblith 當支持。
3. **訓練超參數決定可剪枝性**（但不是低秩）。Zhou, Yang, Chang, Mahoney, ICML
   2023「A Three-regime Model of Network Pruning」：溫度類參數（batch size、學習
   率、訓練長度）對剪枝後效能的影響有 "a sharp transition phenomenon"。
   Na, Mehta, Strubell, Findings of EMNLP 2022「Train Flat, Then Compress」：SAM
   "consistently leads to greater compressibility of parameters compared to
   vanilla Adam"（BERT；剪枝與量化）。兩篇都只讀了摘要。
4. **mixup 與表徵譜**。Verma et al., Manifold Mixup, ICML 2019（搜尋摘要，
   **未開全文**）：manifold mixup 讓各類別表徵的奇異值 "most becoming much smaller"
   ——方向是**壓低**低變異方向。如果我們最後要主張「A1 的低變異方向有功能」，
   這篇會被審稿人拿來反問，要先讀。

RSB 論文本身（Wightman, Touvron, Jégou, arXiv 2110.00476）沒有任何壓縮／剪枝／
低秩的句子；只確認 A1 用 BCE："we adopt the binary cross-entropy (BCE) loss instead
of the typical cross-entropy"，並把 mixup/cutmix 視為多標籤。

## 4. 我們的觀察加了什麼、沒加什麼

### 4.1 加了的（查到的範圍內沒人報告）

1. **同一個變異數門檻 τ、每層各取 k\*(τ)、全部 conv 層同時、不微調、不重建網路**
   的整體 top-1 曲線（τ ∈ {0.9, 0.95, 0.99, 0.999}），在 ResNet-50 V1／ResNet-18／
   VGG-16-BN／ResNet-50 A1 四組權重上，附 random-subspace 對照（同 k、隨機正交
   方向，全部掉到 ≤ 2%）。Zhang 的 Fig. 3 是單層獨立，Garg 的 99.9% 是重訓後；
   ALDS 的 SVD-Energy 基線是權重側。
2. 在 **ResNet-50** 上做（Zhang 2015/2016 只有 SPP-10 與 VGG-16；bottleneck 的
   1×1 擴張層在他們那裡不存在）。
3. 把 k 對照 **r\_max** 而不是 C（τ = 0.999 時 mean k/r\_max = 0.86–0.93，
   mean k/C = 0.63–0.86）。對投影本身這沒有作用（k\*(0.999) ≤ r\_max 自動成立），
   它的價值是讓「保留多少」的讀數跨架構可比。
4. **同架構不同配方的可移除比例不同**：A1 在 τ = 0.999 掉 1.8 點（80.31 → 78.56），
   V1 不掉（76.03 → 76.09）。

### 4.2 沒加的（論文不可以這樣寫）

- 不可寫成「新的壓縮方法」或「我們發現 conv 響應低秩」：Zhang 2015 §3.1 原句
  就是這件事，且他們的 nonlinear／asymmetric 版本比我們的 linear 版本好。
- 不可寫成「不微調也能壓縮是新發現」：Zhang TPAMI Table 7 不微調 4× 只掉 0.9
  top-5；MPC、DDLR、BALF、JRTIP 2023 都以免微調為賣點。
- 不可拿投影實驗宣稱速度或參數量：我們沒有重建層、沒有量 FLOPs，投影是
  「k\* 之外的變異數有沒有被用到」的功能讀數，`projection_check.py` 的 docstring
  已這樣寫，論文措辭要一致。
- 「top-1 不變」的統計解析度：評估集 3,200 張，二項 SE ≈ 0.75 pp（p ≈ 0.76）。
  76.03 → 76.09 是「差異在 0.75 pp 以內」，不是「不變」；A1 的 1.8 pp 約 2.4 SE，
  要用配對檢定（同一張圖投影前後，McNemar）才站得住。規則 9 適用。

### 4.3 配方依賴主張的一個必須先排除的假象

`timm_resnet50.a1_in1k_projection_layers.csv` 裡 **layer1.0.conv3**（C = 256，
r\_max = 64）的 k\*(0.9) = k\*(0.95) = k\*(0.99) = **1**，k\*(0.999) = 16；V1 同一層是
6／10／18／29。也就是 A1 這一層有一個方向吃掉 > 99% 變異（大幅度的單一通道，
BN 之前常見）。A1 在 τ = 0.95 掉到 0.34%（機率水準）、τ = 0.99 只剩 11.0%，而
V1 是 55.2%／73.6%——這個「A1 更敏感」的差距很可能主要是這一層被投影到一維造成
的，不是全網性質。τ = 0.999 的 1.8 點也可能大半來自它（16/64）。

**在寫「可移除比例依配方而異」之前必須做**：(i) 逐層 leave-one-out（只投影一層，
或跳過 layer1.0.conv3）；(ii) 用 6 個 ResNet-50 配方全部跑一遍（規則 10：只有六
配方一致的東西才是 ResNet-50 的性質），目前只有 V1 vs A1 兩點；(iii) 考慮
τ 定義對離群方向的敏感性——這也是規則 4 的另一個面向：變異數比例的 k\* 會被
單一大方向支配，participation ratio 亦然。

## 5. 判定

| 面向 | 判定 |
|---|---|
| 投影操作（響應 PCA → 吸進 W） | **不新**。Zhang et al. CVPR 2015 / TPAMI 2016 Eq. (4)，逐字相同；且是他們的 linear 基線 |
| 用 PCA 能量當秩準則 | **不新**。Zhang §3.4；能量門檻在 ALDS (2021)、BALF (2025) 裡是標準基線 |
| 不微調、全模型 | **不新**。Zhang TPAMI Table 7；MPC 2022、BALF 2025 |
| 統一 τ、每層 k\*(τ)、全層同時的 top-1 曲線＋隨機子空間對照 | **就查到的範圍是新的**，但只是一個小的資料點，屬「k\* 的功能校準」，不是貢獻主張 |
| ResNet-50、r\_max 對照 | 部分新（r\_max 見另一份 note；投影本身不需要它） |
| 可移除比例依訓練配方 | **就查到的範圍是新的**，但 n = 2 且有 §4.3 的層級假象未排除，**目前不得寫成主張**，只能寫成探索性觀察（規則 12） |

整體：**partly novel**——操作與準則要全部歸給 Zhang（與 Garg 的門檻），我們能講的
是「拿量到的 k\*(τ) 當秩、逐層、不微調時，τ = 0.999 在四組權重上保住 top-1（在
0.75 pp 解析度內），τ ≤ 0.99 則否；這證明 k\* 之外的變異數大多不被用到，但 A1 例外」。

## 6. 必引清單（BibTeX；DOI 已用 Crossref 核對）

`garg2019lowEffort`、`wightman2021rsb`、`elmoznino2024highDim`、`han2021rexnet`
已在 `paper/refs.bib`，不重列。以下鍵值為建議，加入前先 grep 有無重複。

```bibtex
@article{zhang2016accelerating,
  author  = {Zhang, Xiangyu and Zou, Jianhua and He, Kaiming and Sun, Jian},
  title   = {Accelerating Very Deep Convolutional Networks for Classification and Detection},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2016},
  volume  = {38},
  number  = {10},
  pages   = {1943--1955},
  doi     = {10.1109/TPAMI.2015.2502579}
}

@inproceedings{zhang2015efficient,
  author    = {Zhang, Xiangyu and Zou, Jianhua and Ming, Xiang and He, Kaiming and Sun, Jian},
  title     = {Efficient and Accurate Approximations of Nonlinear Convolutional Networks},
  booktitle = {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      = {2015},
  pages     = {1984--1992},
  doi       = {10.1109/CVPR.2015.7298809}
}

@inproceedings{denton2014exploiting,
  author    = {Denton, Emily and Zaremba, Wojciech and Bruna, Joan and LeCun, Yann and Fergus, Rob},
  title     = {Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation},
  booktitle = {Advances in Neural Information Processing Systems 27 (NeurIPS)},
  year      = {2014},
  pages     = {1269--1277},
  note      = {arXiv:1404.0736}
}

@inproceedings{jaderberg2014speeding,
  author    = {Jaderberg, Max and Vedaldi, Andrea and Zisserman, Andrew},
  title     = {Speeding up Convolutional Neural Networks with Low Rank Expansions},
  booktitle = {Proc. British Machine Vision Conference (BMVC)},
  year      = {2014},
  doi       = {10.5244/C.28.88}
}

@inproceedings{xue2013restructuring,
  author    = {Xue, Jian and Li, Jinyu and Gong, Yifan},
  title     = {Restructuring of Deep Neural Network Acoustic Models with Singular Value Decomposition},
  booktitle = {Proc. Interspeech},
  year      = {2013},
  pages     = {2365--2369},
  doi       = {10.21437/Interspeech.2013-552}
}

@inproceedings{kim2016compression,
  author    = {Kim, Yong-Deok and Park, Eunhyeok and Yoo, Sungjoo and Choi, Taelim and Yang, Lu and Shin, Dongjun},
  title     = {Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications},
  booktitle = {Int. Conf. Learning Representations (ICLR)},
  year      = {2016},
  note      = {arXiv:1511.06530}
}

@inproceedings{idelbayev2020lowrank,
  author    = {Idelbayev, Yerlan and Carreira-Perpi{\~n}{\'a}n, Miguel {\'A}.},
  title     = {Low-Rank Compression of Neural Nets: Learning the Rank of Each Layer},
  booktitle = {Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
  pages     = {8046--8056},
  doi       = {10.1109/CVPR42600.2020.00807}
}

@inproceedings{liebenwein2021alds,
  author    = {Liebenwein, Lucas and Maalouf, Alaa and Gal, Oren and Feldman, Dan and Rus, Daniela},
  title     = {Compressing Neural Networks: Towards Determining the Optimal Layer-wise Decomposition},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
  year      = {2021},
  note      = {arXiv:2107.11442}
}

@inproceedings{chee2022modelPreserving,
  author    = {Chee, Jerry and Renz, Megan and Damle, Anil and De Sa, Christopher},
  title     = {Model Preserving Compression for Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS)},
  year      = {2022},
  pages     = {38060--38074},
  note      = {arXiv:2108.00065}
}

@article{zhang2026theoretical,
  author  = {Zhang, Shihao and Saab, Rayan},
  title   = {Theoretical Guarantees for Low-Rank Compression of Deep Neural Networks},
  journal = {Applied and Computational Harmonic Analysis},
  year    = {2026},
  volume  = {82},
  pages   = {101837},
  doi     = {10.1016/j.acha.2025.101837}
}

@misc{gonzalezmartinez2025balf,
  author = {Gonz{\'a}lez-Mart{\'i}nez, David},
  title  = {{BALF}: Budgeted Activation-Aware Low-Rank Factorization for Fine-Tuning-Free Model Compression},
  year   = {2025},
  note   = {arXiv:2509.25136}
}

@inproceedings{kornblith2021lossFunctions,
  author    = {Kornblith, Simon and Chen, Ting and Lee, Honglak and Norouzi, Mohammad},
  title     = {Why Do Better Loss Functions Lead to Less Transferable Features?},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
  year      = {2021},
  note      = {arXiv:2010.16402}
}

@inproceedings{kuznedelev2023cap,
  author    = {Kuznedelev, Denis and Kurti{\'c}, Eldar and Frantar, Elias and Alistarh, Dan},
  title     = {{CAP}: Correlation-Aware Pruning for Highly-Accurate Sparse Vision Models},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS)},
  year      = {2023},
  pages     = {28805--28831},
  note      = {arXiv:2210.09223}
}
```

選引（只在寫配方依賴那段時才需要）：

```bibtex
@inproceedings{zhou2023threeRegime,
  author    = {Zhou, Yefan and Yang, Yaoqing and Chang, Arin and Mahoney, Michael W.},
  title     = {A Three-regime Model of Network Pruning},
  booktitle = {Proc. 40th Int. Conf. Machine Learning (ICML), PMLR 202},
  year      = {2023},
  note      = {arXiv:2305.18383}
}

@inproceedings{na2022trainFlat,
  author    = {Na, Clara and Mehta, Sanket Vaibhav and Strubell, Emma},
  title     = {Train Flat, Then Compress: Sharpness-Aware Minimization Learns More Compressible Models},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
  year      = {2022},
  pages     = {4909--4936},
  doi       = {10.18653/v1/2022.findings-emnlp.361}
}

@article{zhang2023jrtip,
  author  = {Zhang, Meng and Liu, Fei and Weng, Dongpeng},
  title   = {Speeding-up and Compression Convolutional Neural Networks by Low-Rank Decomposition Without Fine-Tuning},
  journal = {Journal of Real-Time Image Processing},
  year    = {2023},
  volume  = {20},
  number  = {4},
  pages   = {64},
  doi     = {10.1007/s11554-023-01274-y}
}
```

**論文措辭建議**（投影段落第一句）：
> The projection y ↦ ȳ + U_kU_kᵀ(y − ȳ) is the linear response-reconstruction of
> Zhang et al. [TPAMI 2016, Eq. (4)], i.e. the rank-k factorisation W ≈ U_k(U_kᵀW);
> we use it not as a compression method but as a read-out of whether the variance
> beyond k\*(τ) is used by the network, with the rank of every layer fixed a priori
> by the same τ rather than by a speed budget.

## 7. 查證的侷限

- 讀了全文（pdftotext）：Zhang TPAMI 2016、Zhang CVPR 2015、Denton 2014、
  Jaderberg 2014、Garg 2020、Kim ICLR 2016、Kim & Kyung 2018、ALDS 2021、
  Idelbayev 2020、MPC 2022、DDLR 2021、Zhang & Saab 2025、BALF v3、Kornblith 2021、
  RSB 2021、CAP 2023。引文是從 PDF 轉文字抄的，數學符號（d′、σ、上標）經手工還原，
  引用前對照原 PDF 一次。
- **開不了**：Zhang, Liu, Weng, JRTIP 2023（Springer 與 ACM 都要登入；只有 DOI 與
  作者從 Crossref／Semantic Scholar 取得，方法只知道是 "cosine similarity SVD without
  fine-tuning"）；Kalle et al., NeurIPS 2025「Distribution-Aware Tensor Decomposition
  for Compression of CNNs」（OpenReview 擋機器人，arXiv 找不到）；Xue 2013 只從
  記憶與二手描述；Manifold Mixup 的奇異值句子來自搜尋摘要；Zhou 2023、Na 2022、
  SVD-NAS 2023 只讀摘要。
- 搜尋工具只有 WebSearch／arXiv／Crossref／Semantic Scholar（後者對 SVD-NAS 與
  Kalle 回空）。沒有查 Google Scholar「被引用」清單。
- **沒有找到但不能排除**：2016–2024 之間某篇拿 Zhang 的 linear PCA、固定 99%／99.9%
  能量、在 ResNet-50 上不微調報告 top-1 的短文（workshop 或 arXiv）。建議投稿前
  Google Scholar：`"PCA" "responses" "ResNet-50" "without fine-tuning" low-rank`
  與 `"energy" "99.9%" "singular values" ResNet-50 "no fine-tuning"`。
- 配方依賴（§3）這條線搜尋了六組關鍵字，均無直接命中；"harder to compress" 在
  剪枝界是共識句但沒有同架構跨配方的低秩實驗。這是「找不到」，不是「不存在」。
