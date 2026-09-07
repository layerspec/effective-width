# 正交性兩個命題的新穎性查證（2026-09-07）

對象是規劃中的理論續篇的兩個候選命題：

- **P1（過完備層的正交性極限）**：卷積核活在 d = C_in·k_h·k_w 維；C_out > d 時
  核的精確正交不可能，最佳可達由 frame theory 決定（Welch 界、等角緊框架 ETF、
  Gram 矩陣譜盡量平坦），所以各種正交正則化子（SO、DSO、SRIP、MC）每層有一個
  可算的極小值，在極小值處白噪聲輸入下的輸出共變異譜有特定形狀。
- **P2（與資訊準則等價）**：核正交 W Wᵀ = I 只有在輸入通道共變異 Σ_x 為白時才給
  去相關、等變異的輸出；一般條件是 W Σ_x Wᵀ ∝ I，等價於在總變異固定下最大化
  輸出共變異的高斯熵／participation ratio／有效秩，即最大化 k\*(τ)/r_max。
  所以正交正則化＝在一個真實輸入不滿足的白化假設下做有效寬度最大化。

**結論先講：P1 作為理論命題不新，而且比 `rmax` 那次更不新——最關鍵的部分
（各正則化子在寬矩陣上的極小值集合）已由 Massart (ICPR 2022) 逐條證完；
Welch 界與 frame potential 進卷積層是 Murdock–Lucey (CVPR 2020) 與 Bank–Giryes
(BMVC 2020)。P2 的白化條件是 Huang et al. (AAAI 2018) 的 Theorem 1 逐字，
熵那一半是古典 infomax 與 MCR² (NeurIPS 2020)，「權重正交 ≠ 激活正交」也已有
人量過（Kim, arXiv 2601.00457）。兩個命題剩下能寫的都是**量測**，不是定理。**

以下逐篇列證據，全部來自我自己用 pdftotext 抽出的原文（WebFetch 無法解析
PDF，見 §6）。矩陣約定隨各篇原文，注意有的把濾波器當列、有的當行。

---

## 1. P1：過完備層的正交性極限

### 1.1 「C_out > d 時精確正交不可能」——至少五篇明講

| 文獻 | 原文 |
|---|---|
| Xie, Xiong, Pu, "All You Need is Beyond a Good Init", CVPR 2017, §4 | "generating n orthogonal vectors in d-dimension space which satisfies n > d is ill-posed and, hence, impossible. So one solution is to avoid the fan-ins and fan-outs of kernels violating the principle, say f_in ≥ f_out, in designing structures of networks; another candidate is group-wise orthogonalization proposed by us." |
| Bansal, Chen, Wang, "Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?", NeurIPS 2018, §2 與 §3.1 | "CNN weight matrices cannot exactly lie on a Stiefel manifold as they are either very 'thin' or 'fat' (e.g., WᵀW = I may never happen for an overcomplete 'fat' W due to rank deficiency of its gram matrix)"；"the columns of W could possibly be mutually orthogonal, if and only if W is undercomplete (m ≥ n). For overcomplete W (m < n), its gram matrix WᵀW ∈ R^{n×n} cannot be even close to identity, because its rank is at most m, making ‖WᵀW − I‖²_F a biased minimization objective." |
| Huang et al., "Controllable Orthogonalization in Training DNNs" (ONI), CVPR 2020, §3.4 | "When n > d, the rows of W cannot be orthogonal, because the rank of W is less than/equal to d."；並宣稱 "Our method unifies the row and column orthogonalizations"。 |
| Wang, Chen, Chakraborty, Yu, "Orthogonal Convolutional Neural Networks" (OCNN), CVPR 2020, §3.2 | 針對 doubly block-Toeplitz 矩陣 K："may be a fat matrix (MH'W' ≤ CHW) or a tall matrix (MH'W' > CHW). In either case, we want to regularize the spectrum of K to be uniform. In the fat matrix case, the uniform spectrum requires a row orthogonal convolution, while the tall matrix case requires a column orthogonal convolution, where K is a normalized frame [33] and preserves the norm."；Lemma 1："The row orthogonality and column orthogonality are equivalent in the MSE sense, i.e. ‖KKᵀ − I‖²_F = ‖KᵀK − I'‖²_F + U, where U is a constant." |
| Trockman & Kolter, "Orthogonalizing Convolutional Layers with the Cayley Transform", ICLR 2021, App. A.1 | 只做 semi-orthogonality："Rather, we must resort to enforcing semi-orthogonality"；c_in ≠ c_out 用 zero-padding 處理。 |

Jia et al., "Orthogonal Deep Neural Networks", TPAMI 2021 的主定理也只講到
「非方陣取正交列或正交行」："the optimal bound w.r.t. the degree of isometry is
achieved only when all singular values of each of weight matrices of a DNN are
equal. Among various solutions, the most straightforward one is that all singular
values are equal to 1; in other words, each weight matrix has orthonormal rows or
columns." 沒有 frame、沒有 coherence。BCOP (Li et al. NeurIPS 2019) 與 SOC
(Singla & Feizi 2021) 是運算子正交，文中不處理過完備核（BCOP 反而是用加倍
通道數來繞開表達力問題）。

**所以「不可能」這個事實本身零新穎性，連「必要非充分」（OCNN §3.2："the
kernel orthogonality conditions 7 are necessary but not sufficient conditions
for the orthogonal convolution conditions 3, 6 in general"）都有人講了。**

### 1.2 「各正則化子有可算的極小值」——Massart (ICPR 2022) 已證完主要部分

Estelle Massart, "Orthogonal regularizers in deep learning: how to handle
rectangular matrices?", ICPR 2022, pp. 1294–1299, DOI 10.1109/ICPR56361.2022.9956205。
這篇就是 P1 的核心，而且是 IEEE 會議論文，審稿人查得到。W ∈ R^{n×p}，
n < p（horizontal，即濾波器當行、濾波器數 p 超過維度 n）：

- **Prop. II.1**：‖WᵀW − I_p‖²_F = ‖WWᵀ − I_n‖²_F + p − n。→ SO 與 DSO 等價，
  轉置沒用。（OCNN Lemma 1 是同一件事的卷積版，早一年半。）
- **Cor. II.2**：n < p 時 SO/DSO 的極小值集合 = 所有奇異值皆為 1 的 W，即
  W = U[I_n 0]Vᵀ。→ 極小值是 p − n，由任何 Parseval 緊框架達到。
- **Prop. II.4**：n < p 時 SRIP 的極小值集合 = 所有奇異值落在 [0, √2] 的 W。
  原文結論："using this regularizer on horizontal matrices does not promote
  neither orthonormality of the rows nor of the columns."（極小值 = 1，退化。）
- **Prop. II.5–II.6**：n = p − 1 時 MC 的極小值 = √(n/p)·QM，M 的行是正 n-單體的
  頂點；‖WᵀW − I_p‖_∞ = 1/p。→ 就是 simplex ETF。
- Prop. II.3：Orth-ℓ1 在 n = p − 1 時強迫一行為零（稀疏化）。
- 她自己留下的缺口："relaxing the assumption n = p − 1 makes the analysis
  substantially more complex; we leave this question for further research."

**P1 想做的「每層可算的極小值」，SO/DSO/SRIP 三個已被 Massart 完整解掉，
MC 解到 n = p − 1。** 剩下的 MC 一般情形正是 Welch 界與 ETF 存在性問題
（見 1.3），那是 frame theory 教科書內容。

補一個 SRIP 的實務含意（我自己從 Prop. II.4 推的，Massart 沒有明說）：
ResNet-50 的 20 個 1×1 擴張層若用 SRIP，正則化子在這些層只約束
‖W‖₂ ≤ √2，等於沒在管正交性。Bansal 2018 §3.4 說 "both mutual incoherence
and RIP are well defined for both under-complete and over-complete matrices"
是對的，但「well defined」不等於「有意義」。這一點可以寫，但只是 Massart
的一句推論。

### 1.3 Welch 界、frame potential、ETF 進神經網路——已有兩條線

**Murdock & Lucey, "Dataless Model Selection with the Deep Frame Potential",
CVPR 2020（arXiv 2003.13866）；Murdock, Cazenavette, Lucey, "Reframing Neural
Networks: Deep Structure in Overcomplete Representations", arXiv 2103.05804 v2
(2022)。** 後者 §2.2–2.4：

- 定義 frame potential FP(B) = Σλ²_i = ‖G‖²_F；"Minimizers of the frame potential
  completely characterize the set of all normalized tight frames."；"For tight
  frames, the nonzero eigenvalues are uniform with the value A⁻¹"。
- §2.4.2 推出 Welch 界 μ(B) ≥ √((k d⁻¹ − 1)/(k − 1))（Eq. 18），並說
  "for a fixed input dimensionality d, as the number of parameters increases
  with additional components k, the minimum achievable mutual coherence also
  increases."
- §2.4.3 **卷積框架**：對 p×p 輸入、d 通道、k 個 f×f 濾波器、步幅 s，算出
  結構化的 coherence 下界（Eq. 20），p → ∞ 時退化為多通道非週期界（Eq. 21）：
  "the minimum mutual coherence of a convolutional frame is greater than that
  of a dense frame with the same size due to parameter sharing"。
- 用 minimum deep frame potential 作為與資料無關的模型選擇準則，並展示它與
  validation error 相關（Fig. 10–12）。
- Fig. 6 圖說明講："Mutual coherence is zero in the case of orthogonality,
  which can not occur for overcomplete frames."

**Bank & Giryes, "An ETF view of Dropout regularization", BMVC 2020
（arXiv 1810.06049）。** §3："The Welch bound [70] provides a universal lower
bound on the mean and maximal absolute value of the cross-correlations between
the frame vectors. A frame that achieves the Welch lower bound on the maximal
absolute cross-correlation value is an ETF."；給出 ETF Gram 的閉式（Eq. 4）；
提出以最小化濾波器間 coherence 為正則化子："For convolutional layers, we do
not use their corresponding Toeplitz matrix. Instead, for simplicity, the
coherence between the convolution kernels is minimized."——**這就是核層次的
MC 正則化子＋ETF 目標**。另有 "this is impossible to maintain for over-complete
frames for which all the sub-matrices cannot be unitary."

其他同方向：Cisse et al., "Parseval Networks", ICML 2017（W 為 Parseval
tight frame，d_out ≤ d_in 時 WᵀW ≈ I）；Wang et al., "MMA Regularization",
NeurIPS 2020（Tammes 問題；d ≥ n − 1 時解析解為 max cos = −1/(n−1) 的正單體，
"the analytical solutions for the Tammes problem only exist for some combinations
of n and d"）；Liu et al., MHE, NeurIPS 2018（Thomson 問題）；Ivanitskiy,
Jasper, King, Mixon, "Towards a mathematical theory of superposition",
arXiv 2608.27540 (2026)（Welch–Rankin 界、ETF 在 superposition 的精確恢復門檻）。
最後這篇顯示 frame 社群本身（Mixon 是 ETF 存在性表的作者）已經進場。

### 1.4 「極小值處白噪聲輸入下輸出譜的特定形狀」——兩行線性代數，而且不區分極小值

自己推：濾波器為 W ∈ R^{d×n} 的行（n = C_out > d），輸出 y = Wᵀx，
cov(y) = WᵀΣ_x W，秩 ≤ d = r_max。Σ_x = I 時 cov(y) = WᵀW = Gram。

- SO/DSO 極小值（緊框架，WWᵀ = I_d）：Gram 有 d 個特徵值 = 1、n − d 個 0。
  → 譜在 r_max 維子空間上完全平坦，k\*(τ) = ⌈τ·r_max⌉，k\*/r_max = τ。
- MC 極小值若為 ETF：ETF 也是緊框架，Gram 譜**一樣**是 d 個相等非零值。
- SRIP 極小值：只要求非零特徵值在 [0, 2]，譜形狀不定。

所以「特定形狀」= 「平坦到 r_max」，Reframing §2.2.1 與 OCNN Fig. 1(b)
（"a more ideal uniform spectrum"）已經是這句話；而且 SO 極小值與 MC 極小值
在白輸入下給**相同**的輸出譜，Welch 界／ETF 只影響成對 coherence，不影響譜。
P1 把 Welch 界和譜平坦綁在一起是把兩個不同的目標函數混講——這是 P1 敘述
本身的錯誤，要修。

### 1.5 P1 還剩什麼

1. MC 在一般 n < p 的極小值 = Welch 界，可達當且僅當 (d, C_out) 有 ETF。實
   ETF 需 C_out ≤ d(d+1)/2，存在性查 Fickus & Mixon 的表（"Tables of the
   existence of equiangular tight frames", arXiv 1504.00253；本次未開，憑記憶）。
   這是套用，不是定理。
2. **量測**：在 22 組 checkpoint 上算每個過完備層訓練後核 Gram 的 frame
   potential 與 coherence，對照 Welch 極小與緊框架極小，看訓練後的核離「最佳可達」
   多遠、與 k\*/r_max 有無關係。這沒人做過（Reframing 只算與資料無關的極小值；
   Bank–Giryes 只有 dropout 訓練的 coherence 曲線）。
3. Massart 的 SRIP 退化結論對 ResNet-50 擴張層的含意（1.2 末段）。

審稿人會說："Proposition 1 is Massart (2022) plus Welch (1974); the frame-
potential connection is Murdock & Lucey (2020); the ETF regularizer is Bank &
Giryes (2020). The spectral claim is trivial and does not depend on the Welch
bound."

**判定：P1 不新（not novel）。** 只有 1.5 第 2 點能當作量測貢獻。

---

## 2. P2：正交性與資訊準則的等價

### 2.1 白化條件——Huang et al. AAAI 2018 Theorem 1 逐字

Huang, Liu, Lang, Yu, Wang, Li, "Orthogonal Weight Normalization", AAAI 2018
(arXiv 1709.06079) §2.2.1：

> "Theorem 1. Let s = Wx, where WWᵀ = I and W ∈ R^{n×d}. (1) Assume the mean of
> x is E_x[x] = 0, and covariance matrix of x is cov(x) = σ²I. Then E_s[s] = 0,
> cov(s) = σ²I. (2) If n = d, we have ‖s‖ = ‖x‖. (3) Given the back-propagated
> gradient ∂L/∂s, we have ‖∂L/∂x‖ = ‖∂L/∂s‖."
>
> "The first point of Theorem 1 shows that in each layer of DNNs the weight
> matrix with orthonormality can maintain the activation s to be normalized and
> even de-correlated if the input is whitened."

App. A.1 的證明就是 cov(s) = W cov(x) Wᵀ = σ²WWᵀ。**P2 的「一般條件是
W Σ_x Wᵀ」正是這條證明的第五行**；他們只是沒把 Σ_x ≠ σ²I 的情形反過來當命題講。

同一組人在 ONI (CVPR 2020) §3.5 把它重述成命題並分 n < d / n > d："In
particular, if n < d, property (2) and (3) hold; if n > d, property (1) and (4)
hold."——也就是**過完備時連白輸入下的輸出去相關都不成立**（因為 WWᵀ ≠ I）。
這與 P1 的過完備情形直接相扣，但 ONI 已講。

He, Du, Ma, "Preventing Dimensional Collapse in Self-Supervised Learning via
Orthogonality Regularization", arXiv 2411.00392 (2024) Prop. 1 再用矩陣形式重述
一次："If X̄ = 0 and Σ_X = σ²I, then S̄ = 0 and Σ_S = σ²I"，明寫 "assuming the
input is whitened"，並用 SO/SRIP 畫 ResNet-18 各 block 隱藏特徵的正規化特徵值
衰減，宣稱 OR 減緩衰減（Fig. 3）。這是 P2 的實證版本，已存在。

### 2.2 一般條件 W Σ_x Wᵀ ∝ I 是什麼——白化變換的定義

W Σ Wᵀ = I 就是「W 是 Σ 的白化矩陣」；Kessy, Lewin, Strimmer, "Optimal
whitening and decorrelation", Am. Stat. 2018（憑記憶）把所有解寫成
W = Q Σ^{-1/2}。Huang 等人的 DBN (CVPR 2018) 與 IterNorm (CVPR 2019) 就是在
激活上算 Σ^{-1/2}；OWN 的 App. A.2 甚至也是解 PΣPᵀ = I（只是 Σ 是 proxy
權重的 Gram，不是輸入共變異——寫論文時要把這個同名不同物講清楚）。
Desjardins et al., "Natural Neural Networks", NeurIPS 2015 在每層前放白化層。
**「正交權重＋白化輸入 ⇒ 白化輸出」與「非白輸入需要 Σ-相關的正交」在這些
文獻裡是預設常識。**

過完備時要改寫成：W Σ_x Wᵀ = c·P，P 為秩 d 投影，即 Σ_x^{1/2}Wᵀ 是緊框架。
這是 OCNN「fat 時 row-orthogonal、tall 時 column-orthogonal」在 Σ_x 度量下的
翻譯，仍是重述。

### 2.3 熵／PR／有效秩那一半——古典

- 固定 trace 下特徵值分布的 Shannon 熵在均勻時最大；有效秩 = exp(熵)
  （Roy & Vetterli 2007）；PR ≤ erank（Jensen）。這是定義層次。
- 高斯熵最大化 = infomax（Linsker 1988；Bell & Sejnowski 1995）；去相關 =
  redundancy reduction（Barlow 1961）。
- 現代版：Yu, Chan, You, Song, Ma, "Learning Diverse and Discriminative
  Representations via the Principle of Maximal Coding Rate Reduction", NeurIPS
  2020，目標 R(Z) = ½ log det(I + (d/(mε²)) ZZᵀ)，Theorem 2.1："each subspace
  achieves its maximal dimension ... In addition, the largest d_j − 1 singular
  values of Z*_j are equal"，"features distributed isotropically in each
  subspace"。**「最大化輸出共變異的高斯熵 ⇒ 等特徵值、滿秩」已是 MCR² 的
  定理。**
- SSL 的 Barlow Twins (ICML 2021)、VICReg (ICLR 2022)、W-MSE、Jing et al.
  "Understanding Dimensional Collapse" (ICLR 2022) 全在做「激活共變異 → I」。
- Cogswell et al., DeCov (ICLR 2016) 直接懲罰激活交叉共變異；Rodríguez et al.,
  OrthoReg (ICLR 2017) 改懲罰權重並明說："Regularizing weights is orders of
  magnitude faster than regularizing activations"，且用餘弦相似度使其
  "keep the magnitudes of the detectors unaffected"。權重去相關與激活去相關
  的取捨在 2016–2017 就攤開了。

### 2.4 「權重正交 ≠ 激活正交」的實證——已有人做

Kim, "Geometric Regularization in Mixture-of-Experts: The Disconnect Between
Weights and Activations", arXiv 2601.00457 (2026-01)："We identify a
fundamental disconnect between weight and activation orthogonality: activation
MSO remains ∼1000× higher than weight MSO, with no significant correlation
(r = −0.293, p = 0.523, n=7). This gap arises from non-linear transformations
(SiLU, LayerNorm) and input distribution effects."——n = 7 的 NanoGPT-MoE，
弱，但存在，且已把原因歸到 "input distribution effects"。

### 2.5 P2 敘述裡要修的地方

1. 「等價於最大化 k\*(τ)/r_max」不對。熵、PR、erank 在固定 trace 與秩 ≤ r_max
   下的唯一極大是平坦譜，這三個等價；但 k\*(τ) 是門檻計數，很多非平坦譜也給
   k\*(τ) = r_max（只要前 r_max − 1 個累積不到 τ）。只能說「平坦譜同時極大化
   四者」，不能說「最大化 k\*/r_max ⇔ W Σ Wᵀ ∝ I」。
2. 過完備層 W Σ Wᵀ ∝ I_{C_out} 不可能（秩），要寫成投影（2.2）。
3. 「正交正則化＝白化假設下的有效寬度最大化」這句本身沒錯，但它是 OWN
   Theorem 1 + MCR² Theorem 2.1 的兩步推論，不是新命題。

### 2.6 P2 還剩什麼

- **量測**：在 22 組 ImageNet checkpoint 上同時算每個 conv 層的（a）核 Gram
  平坦度／coherence，（b）輸入通道共變異 Σ_x 的平坦度，（c）輸出 k\*/r_max，
  看 (a) 對 (c) 的解釋力在控制 (b) 後剩多少。He–Du–Ma 只畫特徵值曲線、Kim 只有
  MoE 玩具、OCNN Fig. 1 量的是權重不是激活——**逐層、ImageNet 尺度、把 Σ_x
  放進來的三方對照沒人做**。這和現有 pipeline 幾乎零成本（Σ_x 就是上一層的
  post-activation 共變異，hooks 已在記錄）。
- 若要保留一點理論味：寫成「命題（重述）＋推論」，把 OWN Thm 1 推廣到
  Σ_x ≠ σ²I 與 n > d，明引 Massart 與 ONI，不宣稱新。

審稿人會說："This is Theorem 1 of Huang et al. (2018) read backwards, combined
with the textbook fact that entropy is maximised by a flat spectrum (cf. MCR²).
The claimed equivalence with k\*(τ) is false as stated. Empirical disconnect
already reported by He et al. (2024) and Kim (2026)."

**判定：P2 作為理論命題不新（not novel）；作為量測（Σ_x 控制後的三方對照）
是 partly novel。**

---

## 3. 對續篇的建議

理論續篇不成立。可行的是把兩個命題降為一節「與正交正則化的關係」放進主論文
或短文：

1. 引 Massart 說明 SO/DSO 極小值 = 緊框架、SRIP 在 20/53 擴張層退化；
2. 引 OWN Thm 1 + ONI 說明過完備層即使白輸入也不去相關；
3. 用自己的資料回答一個沒人回答的實證問題：**訓練後的核離緊框架多遠，
   而輸出 k\*/r_max 有多少是核形狀、多少是 Σ_x 形狀決定的。**
4. 順便修正 P2 的 k\* 等價說法（2.5）。

---

## 4. 必引清單（8 篇）

| 文獻 | 一行 |
|---|---|
| Bansal, Chen, Wang, NeurIPS 2018 | SO/DSO/MC/SRIP 定義；明講過完備 Gram 秩虧損、SO 為 biased objective |
| Massart, ICPR 2022 | 寬矩陣上四種正則化子極小值集合的完整刻畫；SRIP 退化；MC 極小為單體 ETF |
| Huang et al., AAAI 2018 (OWN) | Theorem 1：cov(x) = σ²I ⇒ cov(Wx) = σ²I；「de-correlated if the input is whitened」 |
| Huang et al., CVPR 2020 (ONI) | n > d 時列不可能正交；性質 (1)–(4) 依 n ≶ d 分開成立 |
| Wang et al., CVPR 2020 (OCNN) | 核正交必要非充分；fat/tall 用 normalized frame；Lemma 1 列行等價 |
| Murdock & Lucey, CVPR 2020；Murdock et al. 2022 | frame potential、Welch 界進卷積層，含結構化 coherence 下界 |
| Bank & Giryes, BMVC 2020 | ETF 目標、Welch 界、核 coherence 正則化子 |
| Yu et al., NeurIPS 2020 (MCR²) | 最大化 log det 共變異 ⇒ 各子空間等奇異值、滿秩 |

視需要加：Xie et al. CVPR 2017（最早的「n > d 不可能」＋分組正交）；
He, Du, Ma 2024 與 Kim 2026（實證先例）；Cogswell 2016 / Rodríguez 2017
（激活 vs 權重去相關）；Kessy et al. 2018（白化矩陣族）。

---

## 5. 判定

| | 判定 | 理由 |
|---|---|---|
| P1 | **不新** | Massart 2022 已證極小值集合；Welch/ETF 在 NN 層是 Murdock–Lucey、Bank–Giryes；譜平坦與 Welch 界是兩件事，P1 混講 |
| P2 | **不新（理論）／部分新（量測）** | OWN Thm 1 + MCR² Thm 2.1 的推論；k\* 等價說法錯；Σ_x 控制後的逐層三方對照沒人做 |

---

## 6. 查證的侷限

- WebFetch 對 arXiv PDF 全部回傳二進位無法解析；改用 curl 下載後本機
  pdftotext + grep，所以引文是原文但**只讀了 grep 命中的段落附近**，各篇未通讀。
- Semantic Scholar API 兩次查詢回空（疑似限流），沒有拿到 Massart 2022 與
  Bank–Giryes 2020 的被引清單——MC 一般情形的極小值有可能已在它們的後續被解，
  投稿前要用 Google Scholar 查 "cited by"。
- Google Scholar、IEEE Xplore 未直接開；Massart 讀的是作者網站版本，與 IEEE
  正式版可能有頁碼差異。
- Jia et al. TPAMI 只確認了主定理與非方陣措辭，附錄的 Lemma 4.1 之外未讀。
- BCOP、SOC、OPT、MHE 只 grep 了關鍵詞，未讀全文；結論「不處理過完備核」是
  grep 陰性，不是通讀陰性。
- Fickus & Mixon ETF 存在性表、Kessy 2018、Linsker/Bell–Sejnowski、Barlow Twins/
  VICReg 為記憶引用，未在本次開啟。
- 沒有查 OpenReview 上 SRIP／OCNN 的審稿意見，那裡可能已有審稿人提過
  Welch 界。
