# 三篇待確認文獻的查證結果

**2026-09-03。全文已讀，以下取代先前「檢索代理轉述」的版本。**

---

## 1. NORTH\* — Maile et al., AutoML 2022 (arXiv 2202.08539)

*When, where, and how to add new neurons to ANNs*

**結論：比我們原本以為的更接近。三個問題全部是「不利的那個答案」。**

### 有做卷積嗎？有。

架構 **VGG-11 與 WideResNet-28**，資料集 **CIFAR-10 / CIFAR-100**，而且**是加卷積濾波器（channel）**，不是只有全連接神經元。第 4.3 節明講「extend neurogenesis to deep convolutional networks」。

先前標記「是否在卷積濾波器上評估未確認」——現在確認了，是。

### 觸發判準是什麼？

後激活矩陣的 ε-數值秩（Eq. 2）：

    phi_a^ED(f, l) = (1 / M_l) · |{ sigma in SVD(H_l / sqrt(n)) : sigma > epsilon }|

觸發式（Eq. 3）：

    T_act(f, phi_a, l) = min(0, floor( M_l · (phi_a(f,l) − gamma_a · phi_a(f_0,l)) ))

**⚠️ 注意 phi_a^ED 是什麼：逐層激活的數值秩除以層寬 M_l。**
這和我們第一篇量的 k\*/C 是同一個家族的統計量——只是他們用門檻計數奇異值，
我們用累積解釋變異。

**這對我們的影響：**

- **第一篇仍然成立，但引用位置要改。** 他們把這個統計量當成成長過程中的
  **觸發器**，在 CIFAR 上、用 VGG-11/WRN-28；從未報告逐深度的剖面、
  未比較不同指標、未處理文獻矛盾、未做 ImageNet、未做預訓練模型、
  未做初始化對照。我們的貢獻沒有被吃掉。
  但 **NORTH\* 必須引在 Method 一節當作「同一統計量的先前使用」**，
  不能只放在 Related Work 當成一個成長方法帶過。審稿人若自己發現這件事
  而我們沒提，殺傷力很大。
- **第二篇（寬度判準）的 delta 變窄了。** 他們已經有一個基於秩的觸發器。
  我們的差異只剩「他們的 gamma_a 是調出來的、我們的門檻對照經驗虛無分布」。

### 對我們有利的一句

作者自承：**「using a maximum layer size is still needed for these methods to
avoid exploding layer widths, particularly without extensive hyperparameter
tuning」**（第 4.2 節）。

這是現任方法承認自己的判準撐不住——**正是第二篇的開口**。

另外這句也確認了（第 2.2 節）：
「No known works add new neurons that are explicitly more different to existing
neurons than randomly selected weights.」

### 有 deflation 嗎？沒有。

正交只決定**初始化**（function-preserving，fan-out 設為 0），
之後就是一般的 backprop，沒有殘差擬合目標。這一格仍然空著。

---

## 2. GOR — Kurtz, Bar & Giryes, BMVC 2024 (arXiv 2306.10001)

*Group Orthogonalization Regularization*

**結論：確認是相反的事。命名衝突屬實。**

### Eq. 2 逐字

    L_total = L_task + lambda · sum_{l=1}^{L} sum_{i=1}^{N} || W_(i,l)^T W_(i,l) − I ||_F^2

N = 每層的組數，W_(i,l) = 第 l 層第 i 組的攤平濾波器矩陣。

**這是逐組 Gram 懲罰再對組求和——組內正交，完全沒有跨組項。**
全文任何地方都沒有施加或討論跨組正交。

### 摘要的措辭確實誤導

摘要寫「encourages orthonormality **between groups of filters** within the same
layer」，但數學上是各組**內部**的正交。

**還有一層混淆要小心：** 他們自己把「在對應的 normalization group 內做正交」
叫 inter-group，把「跨越 normalization group 的濾波器」叫 intra-group——
**兩者都仍然是逐組懲罰**。他們的術語本身就容易誤導，引用時不要沿用他們的
inter/intra 講法，直接寫數學式。

### 結果

ResNet-110/CIFAR-10 92.73%；ViT-B 在 CIFAR-100/SVHN/Food-101 上 0.3–0.6%；
diffusion LoRA 的 FID（Oxford102 11.01→10.57）；對抗訓練小幅增益。
**典型增益 0.2–0.6%**，計算成本是完整正交化的 1/N。

**對我們的影響：** 支線那篇若要提「組間正交」的構想，必須正面說明
GOR 做的是相反的事、而且報告有效，並回答「為什麼反過來應該也有用」。

---

## 3. PCA-AE 前身 — Pham, Ladjal & Newson (arXiv 1904.01277)

*A PCA-like Autoencoder*

**結論：機制確認是序貫的，但我先前轉述有一處錯誤。**

### 序貫＋凍結：確認

逐字：「We then increase the size of the latent space by 1, while maintaining
the same first component from the previous training: only the second component
is trained.」

**這是真的 deflation 式課程**，先前只是「他人轉述」，現在是原文。

損失 = 重建 ℓ2 + 共變異數項 `sum_{i<k} sum_j z_i^(j) z_k^(j)`
（batch norm 讓平均為零之後），也就是新分量對既有分量去相關。

### ⚠️ 更正：**沒有** PCA 等價定理

我先前把它歸在「PCA 等價」那一族，是錯的。作者明確說
「the autoencoder is a non-linear transformation, **contrary to PCA**」，
全文沒有宣稱也沒有證明線性情形下的等價。

**所以定理領土（ODIN、full-prefix MRL）與機制領土（PCA-AE）是分開的**，
而且兩邊都沒有做卷積濾波器、也都沒有決定 k。這比我先前說的更乾淨。

### 沒有自動維度規則

d_max 由使用者指定：「In all of our experiments, we set d_max of our PCA
autoencoder to the number of parameters used to create the dataset.」

### 只作用在 bottleneck

共變異數項作用在潛在空間的 code 上，**不是卷積濾波器**。

---

## 三句總結

1. **NORTH\* 用的統計量和我們第一篇是同一家族**，必須引在 Method 而非只在
   Related Work；但它沒做剖面、沒做 ImageNet、沒做預訓練模型、沒做初始化對照，
   第一篇的貢獻仍在。
2. **GOR 的命名衝突屬實**，而且它們自己的 inter/intra 術語也會誤導，引用時直接寫式子。
3. **PCA-AE 沒有 PCA 等價定理**（更正我先前的說法），序貫凍結則確認為真。
