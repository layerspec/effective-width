# 寬度理論與訓練前估計的文獻掃描（2026-09-08）

對應 §I 開場主張的查證。擬寫的開場是：

> "every reliable estimate of how much width a layer uses is post hoc; before
> training one can only bound it (by architecture and by data), and no a-priori
> estimator with a guarantee exists."

**結論（先講）：主張可以寫，但必須改寫成三句分開的話，並加兩個限定。**

1. 「最小寬度理論」給的是**函數類的萬有逼近所需寬度**（輸入／輸出維度的
   函數），與訓練後某層輸出共變異數的秩是兩個不同的對象——這句站得住，
   沒有文獻反對。
2. 「訓練前只能給上界」——站得住，但要指名上界是誰的：架構上界 r_max
   （Kim 2018、Han 2021）、資料上界 rank(Σ_patch)（我們的分解），以及
   Daneshmand 2020 給 BN 線性網路的**下界** Ω(√width)。所以正確的說法是
   「訓練前只能給界（上界與、在 BN 下、一個下界），不能給值」。
3. 「沒有帶保證的先驗估計量」——**需要限定**。三個東西會被審稿人拿出來：
   - **Han et al. 2021（ReXNet）**用隨機權重直接量擴張層的 rank ratio 當設計
     指標——這是一個先驗、逐層的秩估計（無保證，且他們量的是非線性之後）。
   - **Daneshmand et al. 2020** 明寫「SGD preserves the initial magnitude of the
     rank to a large extent」，也就是主張初始化秩**能預測**訓練後秩的量級
     （只有最後隱藏層、CIFAR-10、torch.matrix_rank）。
   - **NTK／lazy 極限**（Jacot 2018；Lee 2019）：在該極限下特徵不動，初始化
     秩就是訓練後秩——這是一個有定理的「先驗估計」，只是它描述的正是**不學
     特徵**的體制，而 Yang & Hu 2021 證明 feature learning 體制下沒有這種保證。
   把主張寫成「no a-priori estimator with a guarantee exists **for the width a
   trained layer actually realises, outside the lazy regime**」即可；我們的
   bottleneck 結果（T-R ρ = −0.08）正好是 Daneshmand 式外推的反例。

「post hoc」一詞要小心：Vyas et al. 2023 顯示表徵在不同寬度間一致，意思是
用一個較窄的**已訓練**代理網路可以預測較寬網路的特徵——仍是訓練後估計，
但審稿人可能認為這算「便宜的先驗」。建議在 §I 用 "requires training the
network (or a narrower proxy of it)" 把這條堵住。

---

## 1. 最小寬度／表達力理論

| 文獻 | 精確陳述 | 對象 |
|---|---|---|
| Lu, Pu, Wang, Hu, Wang, NeurIPS 2017 | "width-(n+4) ReLU networks, where n is the input dimension, are universal approximators"；"except for a measure zero set, all functions cannot be approximated by width-n ReLU networks, which exhibits a phase transition" | L¹ 逼近 R^n 上 Lebesgue 可積函數所需的**最小隱藏層寬度**（全網路一致的寬度上限） |
| Hanin & Sellke 2017（arXiv 1710.11278）；期刊版 Hanin, *Mathematics* 2019 | "the minimal width is exactly equal to d_in+1"；"any continuous function on the d_in-dimensional unit cube can be approximated to arbitrary precision by ReLU nets in which all hidden layers have width exactly d_in+1"；多輸出 d_in+d_out | 一致範數、C([0,1]^d_in) |
| Kidger & Lyons, COLT 2020 | 寬度 n+m+2、"any nonaffine continuous function, with a continuous nonzero derivative at some point" 的激活函數，在 C(K;R^m) 稠密；多數激活可降到 n+m+1 | 一致範數、緊集 |
| Park, Yun, Lee, Shin, ICLR 2021 | "the minimum width required for universal approximation of L^p functions is exactly max{d_x+1, d_y}"；ReLU 對一致逼近不成立，加 threshold 激活才成立 | L^p 範數 |
| Cai, ICLR 2023 | "w*_min = max(d_x, d_y)" 為 C-UAP 與 L^p-UAP 共同的下界；leaky-ReLU 在 L^p 下達到（d_x 或 d_y > 1 時） | 任意激活集合 |
| CNN 版本 | 只找到「zero-padding 全卷積網路的 UAP」（SIAM J. Math. Anal. 2024, doi 10.1137/23M1570119；ACHA 2025）與對稱函數的常數通道數結果（殘差全卷積可用常數通道寬；非殘差至少 2 通道、核 ≥ 2）；**沒有**與 r_max 或通道共變異數對應的 CNN 最小寬度結果 | 對稱函數、平移等變函數類 |

**為什麼這不是我們量的東西**（可直接寫進 §II）：這些定理回答的是「要讓
一個寬度受限的網路族在某個函數類上稠密，隱藏層至少要多寬」——它是**存在
性**的、對整個函數類取最壞情形、以 d_x 與 d_y 表示，而且答案（d_x+1 左右）
與實務 CNN 的通道數差兩個數量級。我們量的是**一個已訓練的特定網路**在**一
批特定資料**上，某一層線性輸出的共變異數用了幾個方向。前者是逼近論的下
界，後者是實現量。兩者唯一的交集是「寬度不夠會失去表達力」這個方向，這
不足以預測任何一層的 k*/r_max。

## 2. NTK／無限寬與「有效寬度」

- **Jacot, Gabriel, Hongler, NeurIPS 2018**；**Lee et al., NeurIPS 2019**
  （"Wide neural networks of any depth evolve as linear models under gradient
  descent"）：無限寬極限下網路是線性化模型，各層表徵在訓練中**不變**。這是
  唯一「初始化秩＝訓練後秩」有定理的情形——而它的內容就是「沒有特徵學習」。
- **Yang & Hu, ICML 2021（μP）**："feature learning" 極限與 NTK 極限互斥；
  在 μP 下特徵確實移動。沒有對逐層秩的預測。
- **Vyas, Atanasov, Bordelon, Morwani, Sainathan, Pehlevan, NeurIPS 2023**：
  "structural properties of the models, including internal representations,
  preactivation distributions, edge of stability phenomena, and large learning
  rate effects are consistent across large widths"。意涵：學到的表徵是任務／
  架構族的性質，不隨寬度改變——這**支持**我們「訓練後寬度由學到的核決定」
  的讀法，也意味著寬度可用窄代理網路預測，但仍需訓練。
- **Hanin & Nica, ICLR 2020**：NTK 的有限寬修正以 depth/width 比為尺度；
  「effective width」在這條線上指的是 n/L 這種展開參數，不是逐層秩。
- **Xiao et al., ICML 2018**（mean-field CNN）：隨機 CNN 的訊號傳播與
  dynamical isometry；對象是輸入–輸出 Jacobian 的奇異值，不是通道共變異數。
- 搜尋 "effective width" / "effective number of neurons" 只得到 EFT 式的有限
  寬展開（Roberts–Yaida–Hanin 一系）與 Bayesian NN；**沒有**任何一篇在訓練前
  預測逐層特徵秩。

## 3. 先驗／訓練前的寬度或秩估計

### 3.1 訓練免費的 NAS 代理

| 代理 | 量什麼 | 逐層寬度？ |
|---|---|---|
| NASWOT（Mellor et al., ICML 2021） | "the overlap of activations between datapoints in untrained networks"——整個網路的 ReLU 激活碼 Hamming 核的 log-det | 否，一個網路一個數 |
| Zero-cost proxies（Abdelfattah et al., ICLR 2021） | "use a single-minibatch of data to score a DNN"；synflow、snip、grasp、jacob_cov、fisher；"especially synflow and jacob_cov" 與最終精度相關最高 | 否；synflow 逐參數但求和 |
| TE-NAS（Chen, Gong, Wang, ICLR 2021） | "the condition number of their NTKs, and the number of linear regions in their input space" | 否，全網路 |
| SynFlow（Tanaka et al., NeurIPS 2020） | 逐參數的 synaptic saliency，目的是避免 layer collapse | 逐層有量，但是參數顯著性不是輸出秩 |

**沒有一個訓練免費代理估計某層輸出的有效寬度**；它們都是網路層級的排名分數。

### 3.2 初始化時的秩崩塌

- **Daneshmand, Kohler, Bach, Hofmann, Lucchi, NeurIPS 2020**。對象："the
  rank of the hidden layer activations over a batch"（H_ℓ ∈ R^{d×N}，
  torch.matrix_rank，門檻 σ_max·d·10⁻⁷）。主結果："the rank of linear
  batch-normalized networks scales with their width as Ω(√width)"；反面：
  無正規化線性網路 "provably collapse to rank one, even in the presence of
  residual connections"。Theorem 2 假設 rank(X) = d（**資料滿秩**）與 i.i.d.
  零均值初始化。實驗含 MLP、VGG-19、ResNet-50（附錄 I）在 CIFAR-10。
  **對我們最要緊的一句**："SGD preserves the initial magnitude of the rank to
  a large extent, regardless of the specific network type"（§「Rank through the
  optimization process」）。這是文獻中最接近「初始化秩預測訓練後秩」的主張，
  但只有最後隱藏層、只有量級、只有 CIFAR-10，而且是 BN 有無的對比。我們的
  ResNet-50 bottleneck 結果（訓練後逐層順序與初始化 ρ = −0.08）不與它矛盾
  （量級可保留而順序重排），但顯示它不能推廣到逐層剖面。
- **Dong, Cordonnier, Loukas, ICML 2021**；**Noci et al., NeurIPS 2022**：
  純注意力的秩隨深度雙指數崩塌；Transformer 專屬，對象是 token 矩陣的殘差。
- **Feng et al., NeurIPS 2022**："universal monotonic decreasing property of
  network rank from the basic rules of differential and algebraic composition"；
  "the first empirical analysis of the per-layer behavior of network rank in
  practical settings, i.e., ResNets, deep MLPs, and Transformers on ImageNet"。
  量的是 Jacobian／函數秩沿深度單調下降，無寬度正規化、訓練後量測。
- **Saxe, McClelland, Ganguli, ICLR 2014**：深度線性網路的模態逐一學習——
  訓練後秩由資料 input–output 相關矩陣的奇異值決定，時間上依序出現。這是
  「訓練後秩由資料決定」的理論原型，只對線性網路；與我們「初始化受資料因
  子限制、訓練後受核限制」的敘事要小心區隔：Saxe 的核最終對齊資料，兩者
  在線性極限重合。
- **Han et al., CVPR 2021（ReXNet）**（已在 `rmax-novelty-2026-09-03.md`）：
  隨機權重下量 rank ratio 對 dimension ratio，當作通道數設計準則。**這是文獻
  中唯一的「訓練前逐層秩估計」**，性質是啟發式，無保證，量的是非線性之後。
- **Baker et al., arXiv 2402.06751（2024）**：插入不同寬度的瓶頸，看梯度／
  激活／delta 的秩如何被瓶頸界住——"Layers following a bottleneck have their
  rank bounded by activation rank"。這是**界**，不是估計。

### 3.3 「插瓶頸找飽和點」的實證先驗估計

沒有找到用這個字面說法的論文。最接近的三條線：
1. **Zhang, Zou, He, Sun, TPAMI 2016**（"Accelerating Very Deep Convolutional
   Networks"）：對每層**響應**做 PCA，用累積能量（前 d′ 個特徵值佔比）選每層
   的秩 d′，3,000 張訓練影像。這就是一個逐層的、訓練後的線性有效寬度估計
   （PCA-能量版），比 Ansuini 早三年，而且分母就是通道數 d。**必引**，它是
   我們 k*(τ) 的直接前身；差別是我們用 r_max 正規化、分離 conv 輸出與 block
   輸出、並比較初始化。
2. 低秩展開（Denton 2014；Jaderberg 2014）：逐層降秩到精度掉為止——這是
   「飽和點」做法，但目的在加速，不報告秩本身。
3. Baker 2024（上）：瓶頸插入是分析工具，不是估計器。

## 4. 兩個審稿障礙

### 4.1 Ansuini et al., NeurIPS 2019——「線性估計抓不到」

量什麼："We extracted representations at pooling layers after a convolution or
a block of consecutive convolutions, and at fully connected layers. In the
experiments with ResNets, we extracted the representations after each ResNet
block and the average pooling before the output."（§2）——**全是 block／pooling
輸出**，不是 conv 輸出。

摘要原句："These results can neither be found by linear dimensionality
estimates (e.g., with principal component analysis), nor in representations
that had been artificially linearized."

PC-ID 定義（§3.3）："we defined an 'ad-hoc' estimate of dimensionality by
computing the number of components that should be included to describe 90% of
the variance in the data. In what follows, we call this number PC-ID. We found
PC-ID to be about one or two orders of magnitude larger than the value of the
ID computed with TwoNN. For example, the PC-ID in the last hidden layer of
VGG-16 was ≈ 200 (Fig. 5C, solid red line), while the ID estimated with TwoNN
was ≈ 18 (solid black line)."

他們對差異的解釋："The discrepancy between the ID estimated with TwoNN and
with PCA points to the existence of strong non-linearities in the correlations
between the data, which are not captured by the covariance matrix."

**Fig. 5C 是關鍵反證來源**：他們也畫了**未訓練** VGG-16 的 PC-ID（虛紅線）與
訓練後（實紅線）——原文 "We also computed the PC-ID of the object manifolds
across the layers of VGG-16 on randomly [initialized weights] … (compare solid
and dashed red curves in Fig. 5C)"，接著說 TwoNN ID 訓練前後 "were very
different. While the latter showed the hunchback profile … the former was
remarkably flat." 需要拿 Fig. 5C 看紅線訓練前後差多少——如果紅線幾乎重合，
審稿人會說「線性維度不受訓練影響，Ansuini 早就畫過」，這與我們的
「trained ≠ random, p ≤ 3e-8」直接相關。要在論文裡正面處理：他們的 PC-ID
是 90% 門檻、pooling／block 輸出、未用 r_max 正規化、一組權重。

**「線性張成才是下一層能用的東西」有沒有人講過**：
- **Jazayeri & Ostojic, Curr. Opin. Neurobiol. 2021**："intrinsic dimensionality
  reflects information about the latent variables encoded in collective
  activity while embedding dimensionality reveals the manner in which this
  information is processed"——明確區分內在（流形）維度與嵌入（線性）維度，並
  主張嵌入維度對應下游的線性讀出。這是神經科學側最乾淨的引用。
- Recanatesi et al. 2019（arXiv 1906.00443）、Elmoznino & Bonner 2024 用
  participation ratio（線性）量 CNN，Elmoznino 明說高線性 ED 有利於線性讀出。
- **沒有人**針對 CNN 寫出「下一個卷積是其輸入的線性映射，所以線性張成是
  它能用的全部」這一句。這是我們可以自己講的論證，理由就是分解
  Σ_out = W Σ_patch Wᵀ：下一層的輸出共變異數只依賴輸入的（patch）共變異數，
  流形曲率經 ReLU 才進得來。寫的時候要和規則 4 一致：這不是否定 Ansuini，
  是說兩個量回答不同問題。

### 4.2 核正交 vs 特徵冗餘

**Wang, Chen, Chakraborty, Yu, CVPR 2020（OCNN）**：
- 摘要："the common kernel orthogonality approach, which we show is only
  necessary but not sufficient for ensuring orthogonal convolutions"；§3.x：
  "Obviously, the kernel orthogonality conditions 7 are necessary but not
  sufficient conditions for the orthogonal convolution conditions 3,6 in
  general."
- 他們說的「冗餘」怎麼量：**不是特徵圖共變異數**。§4 "Filter Similarity"：用
  guided back-propagation pattern G ∈ R^{M×CWH}，算 corr(G) 的非對角元素直方
  圖（ImageNet val、ResNet-34 layer 27）；"As the number of channels increases
  with depth from 128 to 512, the curve shifts right and becomes far narrower,
  i.e., more filters become similar."
- 對譜的主張是對 DBT 矩陣 K（線性算子）的奇異值："A standard convolutional
  layer has a long-tailed spectrum. While kernel orthogonality widens the
  spectrum, our OCNN can produce a more ideal uniform spectrum."
- **對我們的意義**：OCNN 講的是算子譜（W 的部分），不是輸出共變異數。用我們
  的分解說得更精確：核正交只約束 WWᵀ；Σ_out = W Σ_patch Wᵀ 的秩與譜還受
  Σ_patch 控制，所以核正交既不必要也不足以讓輸出去相關——除非 Σ_patch 各向
  同性。這一句可放 §II，用來回應「為什麼不直接看核」。

反方向（主張權重正交／去相關能減少特徵冗餘）：
- Bansal, Chen, Wang, NeurIPS 2018："Can We Gain More from Orthogonality
  Regularizations in Training Deep CNNs?"——mutual coherence／RIP 正則化提升
  精度，未量特徵秩。
- Rodríguez et al., ICLR 2017（OrthoReg）："regularizing negatively correlated
  features is an obstacle for effective decorrelation"，局部正交化權重。
- 直接去相關**激活**的一線：Cogswell et al., ICLR 2016（DeCov，最小化隱藏
  激活的交叉共變異數）；Huang et al., CVPR 2018（Decorrelated BN，白化激活）。
  這一線等於承認核層面的正則化不夠，要直接動 Σ_out。

## 5. 訓練後（post hoc）的估計線

- **Ansuini 2019**：TwoNN ID，hunchback；"the ID of the last hidden layer
  predicts classification accuracy on the test set"；"these properties cannot
  be found in networks with random weights or trained on non predictable
  data"（§3.1）。
- **HRank, Lin et al., CVPR 2020**：對象是**單一通道的 H×W 矩陣**的 SVD 秩
  ("where Rank(·) is the rank of a feature map for input image I. We conduct a
  Singular Value Decomposition (SVD) for o_ij(I,:,:)")，與我們的通道共變異數不
  同物。批次穩定性原句："the expectation of ranks generated by a single filter
  is robust to the input images … although different images may have
  different ranks, the variance is negligible. Hence, a small batch of input
  images can be used to accurately estimate the expectation of the feature map
  rank." 實驗用 500 張隨機影像（batch 128）。可引為「秩類量測樣本效率高」的
  先例，但要註明對象不同。
- **Zhang et al., TPAMI 2016**：見 §3.3，逐層響應 PCA 能量選秩——最早的逐層
  線性有效寬度估計。
- Elmoznino & Bonner 2024；Recanatesi 2019：participation ratio 版本。

## 6. 對開場主張的判定

| 子句 | 判定 | 會被拿來反駁的文獻 |
|---|---|---|
| 最小寬度理論界的是函數類逼近所需寬度，不是訓練後某層的實現秩 | **成立** | 無 |
| 訓練前只能給界 | **成立，需補「下界」** | Daneshmand 2020 給 BN 線性網路 Ω(√d) 下界；Kim 2018／Han 2021 上界 |
| 「每一個可靠估計都是 post hoc」 | **需限定** | Han 2021 用隨機權重逐層量秩當設計指標；Daneshmand 2020 "SGD preserves the initial magnitude of the rank"；lazy 極限有定理 |
| 「不存在帶保證的先驗估計量」 | **需限定為「特徵學習體制下」** | Jacot 2018／Lee 2019 在 lazy 極限下就是帶保證的先驗估計 |
| 訓練免費 NAS 代理不估逐層寬度 | **成立** | 無 |

建議改寫（英文）：

> Theory bounds width from above (the reachable rank r_max of the operator,
> and the rank of the input covariance) and, for normalised linear networks,
> from below (Daneshmand et al., 2020); minimum-width theorems bound a
> different quantity altogether, the width sufficient for universal
> approximation. Outside the lazy regime, in which features do not move
> (Jacot et al., 2018; Lee et al., 2019), no estimator predicts the width a
> trained layer actually realises: every such estimate in the literature is
> taken after training (Zhang et al., 2016; Ansuini et al., 2019; Lin et al.,
> 2020), or from a narrower network that was itself trained (Vyas et al.,
> 2023), or from random weights without a guarantee (Han et al., 2021).

## 7. 可放進 Related Work 的段落

**最小寬度理論。** A separate line of work asks how narrow a network can be
while remaining a universal approximator. Lu et al. (2017) showed that
width-(n+4) ReLU networks approximate Lebesgue-integrable functions on R^n and
that width-n networks cannot; Hanin and Sellke (2017; Hanin, 2019) sharpened the
uniform-norm threshold to d_in+1; Kidger and Lyons (2020) extended it to
general activations at width n+m+2; Park et al. (2021) gave the exact L^p
threshold max{d_x+1, d_y}, and Cai (2023) the universal lower bound
max{d_x, d_y}. These results characterise the minimum width that a family of
networks needs to be dense in a function class; they are worst-case over the
class, expressed in the input and output dimensions, and are two orders of
magnitude below the channel counts of practical CNNs. They do not address the
quantity we measure, namely how many directions a particular trained layer
actually uses on a particular data distribution, and no CNN-specific analogue
in terms of channels or kernel support is known to us (the convolutional
universality results of Zhou (2020) and the zero-padding results of 2024–2025
concern translation-equivariant function classes, not channel width).

**訓練免費代理。** Training-free architecture scoring uses quantities available
at initialisation: the overlap of ReLU activation codes across a mini-batch
(Mellor et al., 2021), pruning-at-initialisation saliencies such as SynFlow
(Tanaka et al., 2020; Abdelfattah et al., 2021), and the NTK condition number
together with the number of linear regions (Chen et al., 2021). All of these
produce one score per network for ranking candidates; none estimates the width
a given layer will use after training. The closest a-priori per-layer quantity
is the rank ratio of randomly initialised expansion layers used as a design
heuristic by Han et al. (2021), which carries no guarantee and, as we show for
bottleneck ResNets, does not predict the ordering of layers after training.
Rank-collapse theory (Daneshmand et al., 2020; Saxe et al., 2014) gives lower
bounds and dynamics for linear or batch-normalised networks at initialisation,
and Daneshmand et al. report that SGD largely preserves the initial rank of the
last hidden layer on CIFAR-10; our measurements extend this to all
convolutional layers of ImageNet networks and find that the level is preserved
only up to the kernel factor, while the per-layer profile is not.

## 8. 必引 BibTeX

DOI 標「已查」者為本次查證；其餘為記憶，投稿前用 Crossref 對一次。

```bibtex
@inproceedings{lu2017expressive,
  author={Lu, Zhou and Pu, Hongming and Wang, Feicheng and Hu, Zhiqiang and Wang, Liwei},
  title={The Expressive Power of Neural Networks: A View from the Width},
  booktitle={Advances in Neural Information Processing Systems 30},
  year={2017}, note={arXiv:1709.02540}}

@article{hanin2019universal,
  author={Hanin, Boris},
  title={Universal Function Approximation by Deep Neural Nets with Bounded Width and {ReLU} Activations},
  journal={Mathematics}, volume={7}, number={10}, pages={992}, year={2019},
  doi={10.3390/math7100992}, note={Journal version of Hanin and Sellke, arXiv:1710.11278}}  % 已查

@inproceedings{kidger2020universal,
  author={Kidger, Patrick and Lyons, Terry},
  title={Universal Approximation with Deep Narrow Networks},
  booktitle={Proc. Conf. Learning Theory (COLT)}, series={PMLR}, volume={125}, pages={2306--2327}, year={2020}}

@inproceedings{park2021minimum,
  author={Park, Sejun and Yun, Chulhee and Lee, Jaeho and Shin, Jinwoo},
  title={Minimum Width for Universal Approximation},
  booktitle={Int. Conf. Learning Representations (ICLR)}, year={2021}, note={arXiv:2006.08859}}

@inproceedings{cai2023achieve,
  author={Cai, Yongqiang},
  title={Achieve the Minimum Width of Neural Networks for Universal Approximation},
  booktitle={Int. Conf. Learning Representations (ICLR)}, year={2023}, note={arXiv:2209.11395}}

@inproceedings{jacot2018ntk,
  author={Jacot, Arthur and Gabriel, Franck and Hongler, Cl{\'e}ment},
  title={Neural Tangent Kernel: Convergence and Generalization in Neural Networks},
  booktitle={Advances in Neural Information Processing Systems 31}, year={2018}}

@article{lee2020wide,
  author={Lee, Jaehoon and Xiao, Lechao and Schoenholz, Samuel S. and Bahri, Yasaman and Novak, Roman and Sohl-Dickstein, Jascha and Pennington, Jeffrey},
  title={Wide Neural Networks of Any Depth Evolve as Linear Models under Gradient Descent},
  journal={J. Stat. Mech.: Theory Exp.}, volume={2020}, pages={124002}, year={2020},
  doi={10.1088/1742-5468/abc62b}, note={Conference version: NeurIPS 2019}}

@inproceedings{yang2021feature,
  author={Yang, Greg and Hu, Edward J.},
  title={Tensor Programs {IV}: Feature Learning in Infinite-Width Neural Networks},
  booktitle={Proc. Int. Conf. Machine Learning (ICML)}, series={PMLR}, volume={139}, pages={11727--11737}, year={2021}}

@inproceedings{vyas2023feature,
  author={Vyas, Nikhil and Atanasov, Alexander and Bordelon, Blake and Morwani, Depen and Sainathan, Sabarish and Pehlevan, Cengiz},
  title={Feature-Learning Networks Are Consistent Across Widths at Realistic Scales},
  booktitle={Advances in Neural Information Processing Systems 36}, year={2023}, note={arXiv:2305.18411}}

@inproceedings{hanin2020finite,
  author={Hanin, Boris and Nica, Mihai},
  title={Finite Depth and Width Corrections to the Neural Tangent Kernel},
  booktitle={Int. Conf. Learning Representations (ICLR)}, year={2020}, note={arXiv:1909.05989}}

@inproceedings{xiao2018dynamical,
  author={Xiao, Lechao and Bahri, Yasaman and Sohl-Dickstein, Jascha and Schoenholz, Samuel S. and Pennington, Jeffrey},
  title={Dynamical Isometry and a Mean Field Theory of {CNNs}: How to Train 10,000-Layer Vanilla Convolutional Neural Networks},
  booktitle={Proc. Int. Conf. Machine Learning (ICML)}, series={PMLR}, volume={80}, pages={5393--5402}, year={2018}}

@inproceedings{mellor2021naswot,
  author={Mellor, Joe and Turner, Jack and Storkey, Amos and Crowley, Elliot J.},
  title={Neural Architecture Search without Training},
  booktitle={Proc. Int. Conf. Machine Learning (ICML)}, series={PMLR}, volume={139}, pages={7588--7598}, year={2021}}

@inproceedings{abdelfattah2021zero,
  author={Abdelfattah, Mohamed S. and Mehrotra, Abhinav and Dudziak, {\L}ukasz and Lane, Nicholas D.},
  title={Zero-Cost Proxies for Lightweight {NAS}},
  booktitle={Int. Conf. Learning Representations (ICLR)}, year={2021}, note={arXiv:2101.08134}}

@inproceedings{chen2021tenas,
  author={Chen, Wuyang and Gong, Xinyu and Wang, Zhangyang},
  title={Neural Architecture Search on {ImageNet} in Four {GPU} Hours: A Theoretically Inspired Perspective},
  booktitle={Int. Conf. Learning Representations (ICLR)}, year={2021}, note={arXiv:2102.11535}}

@inproceedings{tanaka2020synflow,
  author={Tanaka, Hidenori and Kunin, Daniel and Yamins, Daniel L. K. and Ganguli, Surya},
  title={Pruning Neural Networks without Any Data by Iteratively Conserving Synaptic Flow},
  booktitle={Advances in Neural Information Processing Systems 33}, year={2020}, note={arXiv:2006.05467}}

@inproceedings{daneshmand2020bn,
  author={Daneshmand, Hadi and Kohler, Jonas and Bach, Francis and Hofmann, Thomas and Lucchi, Aurelien},
  title={Batch Normalization Provably Avoids Rank Collapse for Randomly Initialised Deep Networks},
  booktitle={Advances in Neural Information Processing Systems 33}, year={2020}, note={arXiv:2003.01652}}

@inproceedings{dong2021attention,
  author={Dong, Yihe and Cordonnier, Jean-Baptiste and Loukas, Andreas},
  title={Attention Is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth},
  booktitle={Proc. Int. Conf. Machine Learning (ICML)}, series={PMLR}, volume={139}, pages={2793--2803}, year={2021}}

@inproceedings{noci2022signal,
  author={Noci, Lorenzo and Anagnostidis, Sotiris and Biggio, Luca and Orvieto, Antonio and Singh, Sidak Pal and Lucchi, Aurelien},
  title={Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse},
  booktitle={Advances in Neural Information Processing Systems 35}, year={2022}, note={arXiv:2206.03126}}

@inproceedings{feng2022rank,
  author={Feng, Ruili and Zheng, Kecheng and Huang, Yukun and Zhao, Deli and Jordan, Michael and Zha, Zheng-Jun},
  title={Rank Diminishing in Deep Neural Networks},
  booktitle={Advances in Neural Information Processing Systems 35}, year={2022}, note={arXiv:2206.06072}}

@inproceedings{saxe2014exact,
  author={Saxe, Andrew M. and McClelland, James L. and Ganguli, Surya},
  title={Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks},
  booktitle={Int. Conf. Learning Representations (ICLR)}, year={2014}, note={arXiv:1312.6120}}

@inproceedings{ansuini2019intrinsic,
  author={Ansuini, Alessio and Laio, Alessandro and Macke, Jakob H. and Zoccolan, Davide},
  title={Intrinsic Dimension of Data Representations in Deep Neural Networks},
  booktitle={Advances in Neural Information Processing Systems 32}, year={2019}, note={arXiv:1905.12784}}

@inproceedings{lin2020hrank,
  author={Lin, Mingbao and Ji, Rongrong and Wang, Yan and Zhang, Yichen and Zhang, Baochang and Tian, Yonghong and Shao, Ling},
  title={{HRank}: Filter Pruning Using High-Rank Feature Map},
  booktitle={Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)}, pages={1526--1535}, year={2020},
  doi={10.1109/CVPR42600.2020.00160}}

@inproceedings{wang2020ocnn,
  author={Wang, Jiayun and Chen, Yubei and Chakraborty, Rudrasis and Yu, Stella X.},
  title={Orthogonal Convolutional Neural Networks},
  booktitle={Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)}, pages={11502--11512}, year={2020},
  doi={10.1109/CVPR42600.2020.01152}}

@inproceedings{bansal2018orthogonality,
  author={Bansal, Nitin and Chen, Xiaohan and Wang, Zhangyang},
  title={Can We Gain More from Orthogonality Regularizations in Training Deep {CNNs}?},
  booktitle={Advances in Neural Information Processing Systems 31}, year={2018}, note={arXiv:1810.09102}}

@inproceedings{rodriguez2017orthoreg,
  author={Rodr{\'\i}guez, Pau and Gonz{\`a}lez, Jordi and Cucurull, Guillem and Gonfaus, Josep M. and Roca, Xavier},
  title={Regularizing {CNNs} with Locally Constrained Decorrelations},
  booktitle={Int. Conf. Learning Representations (ICLR)}, year={2017}, note={arXiv:1611.01967}}

@inproceedings{cogswell2016decov,
  author={Cogswell, Michael and Ahmed, Faruk and Girshick, Ross and Zitnick, Larry and Batra, Dhruv},
  title={Reducing Overfitting in Deep Networks by Decorrelating Representations},
  booktitle={Int. Conf. Learning Representations (ICLR)}, year={2016}, note={arXiv:1511.06068}}

@inproceedings{huang2018dbn,
  author={Huang, Lei and Yang, Dawei and Lang, Bo and Deng, Jia},
  title={Decorrelated Batch Normalization},
  booktitle={Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)}, pages={791--800}, year={2018},
  doi={10.1109/CVPR.2018.00089}}

@article{zhang2016accelerating,
  author={Zhang, Xiangyu and Zou, Jianhua and He, Kaiming and Sun, Jian},
  title={Accelerating Very Deep Convolutional Networks for Classification and Detection},
  journal={IEEE Trans. Pattern Anal. Mach. Intell.}, volume={38}, number={10}, pages={1943--1955}, year={2016},
  doi={10.1109/TPAMI.2015.2502579}}  % 已查

@article{jazayeri2021interpreting,
  author={Jazayeri, Mehrdad and Ostojic, Srdjan},
  title={Interpreting Neural Computations by Examining Intrinsic and Embedding Dimensionality of Neural Activity},
  journal={Current Opinion in Neurobiology}, volume={70}, pages={113--120}, year={2021},
  doi={10.1016/j.conb.2021.08.002}}  % 已查

@misc{baker2024lowrank,
  author={Baker, Nathaniel and others},
  title={Low-Rank Learning by Design: The Role of Network Architecture and Activation Linearity in Gradient Rank Collapse},
  year={2024}, note={arXiv:2402.06751; 作者清單待補}}

@misc{recanatesi2019dimensionality,
  author={Recanatesi, Stefano and Farrell, Matthew and Advani, Madhu and Moore, Timothy and Lajoie, Guillaume and Shea-Brown, Eric},
  title={Dimensionality Compression and Expansion in Deep Neural Networks},
  year={2019}, note={arXiv:1906.00443}}
```

已在 `refs.bib` 的 `han2021rexnet`、`kim2018rank1`、Elmoznino & Bonner 2024
不重列。

## 9. 查證的侷限

- 只用 WebSearch／arXiv；**Semantic Scholar 與 Google Scholar 本次未查**（上次
  429），被引清單不完整。
- **打不開**：CVF Open Access 的 OCNN PDF（HTTP 403）——改用 arXiv 1911.12207
  的 PDF，本機 pdftotext 抽字，引文以 arXiv 版為準，頁碼／式號可能與 CVPR
  版差一。WebFetch 對 PDF 一律讀不到內文，Ansuini、HRank、Daneshmand、OCNN
  四篇都是存檔後本機抽字；Lu 2017、Hanin & Sellke、Kidger & Lyons、Park
  2021、Cai 2023、Feng 2022 **只讀了 arXiv 摘要頁**，定理的精確條件（範數、
  定義域、激活）未逐一核對全文。
- Jacot 2018、Lee 2019、Yang & Hu 2021、Saxe 2014、Dong 2021、Noci 2022、
  Tanaka 2020 本次**未開啟**，敘述來自記憶；它們對本主張只作背景，不是反證
  來源。
- Ansuini Fig. 5C 的訓練前／後 PC-ID 紅線差距**沒有看圖**，只有文字；投稿前
  要打開 PDF 看那張圖，因為它直接關係到 "trained ≠ random" 是否已被畫過。
- 「插瓶頸找飽和點」的原始論文**沒有找到**；若作者記得的是某篇特定文獻，
  請補題名再查。目前以 Zhang 2016（PCA 能量選秩）與 Baker 2024（瓶頸界秩）
  代替。
- CNN 專屬的最小寬度結果：只找到全卷積 UAP（SIAM JMA 2024、ACHA 2025）與
  對稱函數的常數通道數結果，兩篇都沒開全文；沒有以通道數表示的最小寬度
  定理，但不能排除 2025–2026 有新文。
- DOI：只有三個是本次查到的（標「已查」）；CVPR 兩個 DOI 與 PMLR 卷頁碼來自
  記憶，投稿前用 Crossref 對。
