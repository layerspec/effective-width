# 「核的增益對齊輸入主方向」的先行工作（2026-09-29 傍晚）

起因：查 Th2／Th5 時碰到 Maennel et al. 2020，作者要我把它的「Deeper layers」一節讀完。讀完後順著線索又查到兩篇更接近的。
三篇都抓 PDF 讀原文。**三篇目前都不在 `refs.bib`，正文、補充也沒有提。**

論文現在的主張（`main.tex` 910–933 行）：W ＝ Σ s_i u_i v_iᵀ，ρ_align 是「核的增益 s_i²」與「該方向上的輸入變異 v_iᵀΣ_patch v_i」
在保留的右奇異方向上的 Spearman 相關；19 個 ImageNet 權重的網路中位數 +0.91，48 個訓練後網路中 43 個 p < 0.01，20 個隨機初始化全部 |ρ| < 0.1；
epoch 7–15 到位；歸屬寫的是「Saxe et al. 對線性層導出的行為，這裡在訓練後的卷積上量到」。

---

## 1. Maennel et al., NeurIPS 2020（arXiv 2006.10455）

作者：Maennel, Alabdulmohsin, Tolstikhin, Baldock, Bousquet, Gelly, Keysers（Google Research）。

| 項目 | 原文 |
|---|---|
| 定義 | Def. 1：A 與 B「aligned」＝ B 的每個特徵空間包含於 A 的某個特徵空間（一般位置下即共用特徵向量）。Def. 2：misalignment 分數 M(A, B)。 |
| 命題 1 | 輸入 N(0, Σ_x)、各向同性初始化、**隨機標籤**、SGD：第一層權重的二階動差 Σ_w ＝ E[wwᵀ] 與 Σ_x 對齊。證明用對稱性。 |
| 優先權 | 「To the best of our knowledge, the alignment effect has not been established in the literature before.」 |
| 真實標籤 | 「we also observe them empirically for real labels」；理由是小 patch 幾乎與標籤獨立。 |
| 量了哪一層 | **只有第一層**：圖 2 是兩層網路（256 個卷積濾波器）在 CIFAR10 的第一層；圖 3 是 70 個 WRN-28-4 的第一層濾波器。 |
| 轉移函數 f(σ) | Def. 3：f(σ_i) ＝ √(v_iᵀ Σ_w v_i)，v_i 是**資料**的特徵向量。形狀「increasing, then decreasing」：增益隨資料變異上升，但最大的一兩個特徵值被壓低。真實影像、重疊 patch（stride 1）時偏離這個形狀（附錄 E.2）。 |
| **§2.5 Deeper layers** | **沒有量深層的對齊。** 做的是建構：把前面各層的輸出當輸入資料，取其共變異數的前 d 個特徵向量乘上假定的 τ_i 當作該層的濾波器，逐層做下去；在一個三層卷積的小網路（每層 16 個濾波器，CIFAR10）上，這樣的初始化重現了「隨機標籤預訓練」的正向遷移（表 1–7）。用最小特徵值的特徵向量則比隨機初始化還差（表 5）。 |

**與我們的關係。** 他們的 f(σ)² ＝ v_iᵀΣ_w v_i，而 Σ_w ＝ WᵀW／C，所以 f(σ_i)² ＝ ‖Wv_i‖²／C，就是**核在資料主方向上的增益**。
我們的 ρ_align 用的是核自己的右奇異方向、量那個方向上的輸入變異；他們用資料的特徵方向、量那個方向上的核增益。
兩者是同一個關係的兩種讀法（對齊成立時兩組基底相同）。所以：

- **第一層的對齊、以及「增益隨輸入變異上升」：Maennel 已有（命題＋量測）。**
- **深層：他們假設對齊成立並拿來建構初始化，沒有在訓練後的網路上量。** 我們逐層量（每層對它自己的輸入 patch 共變異數），
  48 個訓練後網路、含 19 個 ImageNet 公開權重、真實標籤、隨機對照、到位時間——這些是他們沒有的。
- 他們的 f 在頂端下降（最大特徵值被壓低）；我們報的是單一個 Spearman 係數，看不出頂端。**要查我們的資料在頂端是否也下降**
  （逐方向的 s_i² 與 v_iᵀΣv_i 目前沒有存檔，`*_decompose.csv` 只有每層一個 ρ_align；要重跑 decompose 並存逐方向的值）。

## 2. Beaglehole, Radhakrishnan, Pandit & Belkin（arXiv 2309.00570，2023）；Radhakrishnan et al., Science 2024

Convolutional Neural Feature Ansatz（CNFA）：「covariances of filters in any convolutional layer are proportional to the average gradient
outer product (AGOP) taken with respect to patches of the input to that layer」。

- 量測：「high average Pearson correlation (in most cases > .9) between AGOP on patches and the covariance of filters **across all layers**
  of pre-trained convolutional networks on ImageNet」——AlexNet、全部 VGG、全部 ResNet（ResNet 只算每個 block 的第一層）。
- 理論：「We additionally prove that the CNFA holds for one step of gradient descent for deep convolutional networks.」
- 全文沒有出現輸入 patch 的共變異數或主成分（grep 無命中）。

**這是最接近、也最有名的先行工作**（全連接版登在 Science）。他們刻畫的是同一個矩陣 WᵀW，在同一類公開權重上、每一層都量了，
只是比較對象是 patch 的 AGOP（梯度外積），不是 patch 的共變異數。2026 年的審稿人幾乎一定知道 NFA，會問：
**「你的對齊是不是 NFA 的影子？」** 這一問我們現在答不出來，因為沒有量過 AGOP。

可能的關係（未驗證，只是推理）：梯度更新 ΔW ∝ δxᵀ，所以 WᵀW 累積的是「以回傳訊號加權的 xxᵀ」；AGOP 與 Σ_patch 都是這個加權二階動差
的不同讀法。若如此，與 Σ_patch 的對齊是 NFA 的粗略版本。

## 3. Boix-Adserà, Mallinar & Simon（arXiv 2507.05644，2025）：Features at Convergence Theorem（FACT）

用一階最適條件（含 weight decay）導出收斂時 WᵀW 滿足的等式，「explains why the NFA holds in most settings」。

**這同時蓋掉了證明總帳的 Th10**（權重衰減下核的列空間收斂到資料支撐）：有 weight decay 時的一階條件給 W ＝ −(1/λ) E[∇_z L · xᵀ]，
W 的列空間落在輸入張成的空間內，是 FACT 的直接推論。Th10 不必自己證，引用即可。

## 4. 判定

| 我們的陳述 | 先行工作 | 判定 |
|---|---|---|
| 訓練後核的高增益方向與輸入的高變異方向對齊 | Maennel 2020（第一層，命題＋量測）；CNFA 2023（每一層，對 AGOP） | **現象已知**；我們的是「對 Σ_patch、逐層、公開權重、附隨機對照」的量測 |
| 對齊在 epoch 7–15 到位、早於剖面與表示 | 未見 | 我們的（時間過程） |
| 對齊使輸出譜比等距反事實更陡，所以寬度低於正交化的預測 | 未見 | 我們的（對齊 → 寬度的後果；ONI 驗證） |
| inverted residual 家族對齊弱（+0.11 到 +0.61） | 未見；但這兩個家族有 5 個權重量測時正規化用錯（plan §9.20），要等更正 | 待更正後重判 |
| 歸屬「Saxe 的深線性預測」 | — | **不夠**：要加 Maennel 2020 與 CNFA／NFA；現在的寫法會被認為不知道這兩篇 |

## 5. 要做的

| # | 項目 | 成本 | 說明 |
|---|---|---|---|
| L1 | 引用三篇並改寫 `sec:decompose` 的歸屬句、§I 貢獻第 3 條、摘要「whose gain aligns…」一句 | 改稿時 | 摘要與 §I 本來就等框架審稿 |
| L2 | **量 AGOP，回答「是不是 NFA 的影子」**：在 VGG-16、ResNet-18（ImageNet 公開權重）每層算 patch-AGOP，與 WᵀW、Σ_patch 三者兩兩比較（Pearson 與我們的 ρ_align 型統計量） | Mac，約一天（要反向傳播） | **先寫預登記**。預測：AGOP 對 WᵀW 的相關 > Σ_patch 對 WᵀW 的相關（NFA 是較精確的說法）；若如此，論文寫「對齊是 NFA 在二階統計上的投影，我們用它是因為它不需要標籤與梯度」 |
| L3 | 重跑 decompose 存逐方向的 (s_i², v_iᵀΣv_i)，看頂端是否如 Maennel 的 f 那樣下降 | Mac，半天 | 探索性 |
| L4 | `refs.bib` 加 Maennel 2020、Beaglehole 2023、Radhakrishnan 2024（Science）、Boix-Adserà 2025；照慣例先核 | 1 小時 | |

## 6. 對證明總帳的更新

- **Th10**：由「可證未證」改為「**已知**（FACT 2025 的一階條件；引用）」。
- **Th11**：由「只有實證」改為「現象已知（Maennel 2020 第一層；CNFA 2023 對 AGOP）；我們的是逐層對 Σ_patch 的量測、時間過程、對寬度的後果」。
- 五條原訂要補的證明（Th2、Th5、Th8、Th9、Th10）現在四條確認已知，剩 Th8（ReLU）未查。
