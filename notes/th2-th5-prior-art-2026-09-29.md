# Th2、Th5 文獻查核（2026-09-29 下午）

對象：`notes/proofs-2026-09-29.md` 的 Th2（樣本特徵值前 k 和的偏差方向與上界）與 Th5／Th5′（初始化寬度的集中）。
方法同 Th9：WebSearch 找候選，抓 PDF 用 pdftotext 讀原文。讀了 8 篇（兩篇 kernel PCA、Reiss–Wahl、Koltchinskii–Lounici、
非線性隨機矩陣的集中、Maennel 2020，另兩篇只到摘要）。Cacoullos & Olkin 1965 付費牆，只確認題目與出處。

**結論先講：兩條都不是新數學。Th2 的方向與 √(k/n) 上界在 kernel PCA 文獻 2005–2007 年就有，而且有比我證的更緊的界；
Th5 其實與 Th2 是同一個定理——高斯核的 WΣWᵀ 就是「以寬度 C 為樣本數」的樣本共變異數，同一批文獻直接適用。
我在證明筆記裡寫的兩處「極限」（更緊的界證不出來）是我沒查文獻就下的判斷，是錯的，已更正。
查核另外抓到一篇必須引而沒引的：Maennel et al. (NeurIPS 2020) 已證第一層權重與資料主成分對齊。**

---

## 1. Th2：已有的結果

| 文獻 | 原文說了什麼 | 對應 Th2 的哪一部分 |
|---|---|---|
| Shawe-Taylor, Williams, Cristianini & Kandola, IEEE Trans. Inf. Theory 51(7):2510–2522, 2005 | 「the empirical partial sum initially overestimates the process partial sum」；尾和則低估；「One of the results of this paper will be bounds on the degree of overestimation for these partial sums in a fully general setting」。證明用 Courant–Fischer 的變分刻畫取期望——與我的 Ky Fan＋Jensen 是同一個論證。Thm 7：前 k 個（或後面的）特徵值之和的集中不等式（McDiarmid） | **(1) 方向：已有**，一般分布、不需譜隙；集中也已有 |
| Blanchard, Bousquet & Zwald, Machine Learning 66:259–294, 2007 | Thm 3.1：重建誤差（＝尾和）的界，量級 √(d/n · tr(C₂ − C₁⊗C₁))；**Thm 3.5：資料以樣本均值置中時同樣成立**；Thm 3.2（excess bound）與 Thm 3.4（relative bound）用局部 Rademacher 平均，收斂率「typically faster than n^{−1/2}」，依賴特徵值衰減；並明說結果可轉成「partial sums of the biggest or smallest eigenvalues」的界 | **(2) √(k/n) 上界：已有**，連形式都一樣（tr(C₂ − C₁⊗C₁) 就是 E‖xxᵀ − Σ‖_F²）；**置中的情形 (iii) 也已有**；**而且有更緊的界** |
| Reiß & Wahl, Ann. Statist. 48(2), 2020（arXiv 1609.03779） | PCA 重建誤差的非漸近上界，「oracle inequalities under mild eigenvalue conditions」；明確處理指數與**多項式衰減**的譜 | 冪律譜下的緊界：已有 |
| Cacoullos & Olkin, Biometrika 52:87–94, 1965 | 題目「On the bias of functions of characteristic roots of a random matrix」；全文未讀 | 偏差方向的古典來源（待讀） |
| Lawley 1956（已在 refs.bib） | 一階展開，需譜隙 | 正文命題 1 現在用的 |

**我的筆記寫錯的地方**：Th2 的「極限」一段說「更緊的、不需譜隙的界我們沒有證出來（障礙：局部譜密度）」。事實是 Blanchard et al. 的 relative bound
與 Reiß–Wahl 已經給了依賴譜衰減的更緊的界。正確的寫法是「引用，並在我們的譜上算出它們的數值」。

**沒有找到的**：把這些結果用在網路激活的 k*(τ) 上、當作樣本門檻的依據。維度估計這一邊查到的是 participation ratio 的偏差
（Chun et al. 2026，已在 refs.bib）與皮質資料的方法學批評（Pospisil & Pillow 2025，已在 refs.bib）。搜尋有限，信心中等。

**Th2 還剩什麼是我們的**：(3) 比例的界與 (4) 對 k*(τ) 的後果是兩行推論；逐位置抽樣下 Σ̄ 與 Σ 的差 ≤ 1/m 是簡單計算。
都是應用，不是定理。

## 2. Th5：已有的結果

**關鍵觀察（我證的時候沒有看出來）**：W ＝ sG、G 為 C×d 高斯時，M ＝ WΣ^{1/2} 的 C 列是 C 個獨立的 N(0, s²Σ) 樣本，
MᵀM／C 是這 C 個樣本的樣本共變異數，而 WΣWᵀ ＝ MMᵀ 與 MᵀM 有相同的非零特徵值。所以
**「初始化時一層的輸出譜」＝「以寬度 C 為樣本數、對輸入譜做一次抽樣」**，Th5 就是 Th2 換了角色：

| | Th2 | Th5 |
|---|---|---|
| 母體共變異數 | 激活的 Σ | 輸入 patch 的 Σ |
| 樣本數 | n（影像 × 位置） | C（通道數） |
| 隨機性來自 | 影像 | 權重 |

因此 §1 的每一篇都直接適用於 Th5。另外：

| 文獻 | 原文 | 對應 |
|---|---|---|
| Koltchinskii & Lounici, Bernoulli 23(1), 2017（arXiv 1405.2468） | 高斯樣本：E‖Σ̂ − Σ‖ ≍ ‖Σ‖(√(r(Σ)/n) ∨ r(Σ)/n)，r(Σ) ＝ tr Σ/‖Σ‖；並有繞期望或中位數的集中不等式 | 我的 Th5 裡出現的 sr(Σ) 就是他們的 effective rank；算子範數的集中已有 |
| Shawe-Taylor et al. 2005, Thm 7 | 前 k／後面特徵值之和的集中 | Th5 的頭部與尾部 |
| Reiß & Wahl 2020 | 多項式衰減下的緊界 | Th5′ 想做的事 |
| Adamczak 等（arXiv 2507.07625，2025） | 非線性隨機矩陣的集中不等式，推論包含神經網路 conjugate kernel 的線性譜統計量 | 對應 **Th8**（ReLU 之後）；要讀，可能已涵蓋 |

**Th5′ 的 Poincaré 變異數界**（Var(R_k) ≤ 4c_P s² E tr(M_⊥ΣM_⊥ᵀ)）我沒有在讀過的文獻裡看到完全相同的式子，但它是標準工具的直接應用，
而且 Reiß–Wahl 的界很可能更緊。不當作貢獻。

**我的筆記寫錯的地方**：Th5 的「極限」說「非漸近、又緊到幾個通道的界我們沒有證出來」。應改為「Reiß–Wahl／Blanchard 的界待在我們的譜上算出數值比較」。

**「寬度就是樣本數」這個觀察有用**：它把樣本門檻（n/C ≥ 50）與初始化剖面放進同一個框架——n 個樣本估 C 維共變異數，與 C 個通道
抽樣 d 維輸入譜，是同一個統計問題。這不是新數學（random features／conjugate kernel 文獻一直這樣看），但對論文的敘述有幫助：
S2「初始化可算」可以寫成「a random layer is a C-sample PCA of its input」，Th2 的上偏告訴我們初始化的輸出譜比輸入譜更集中。

## 3. 查核順帶抓到的：Maennel et al. 2020（未引用）

Maennel, Alabdulmohsin, Tolstikhin, Baldock, Bousquet, Gelly & Keysers, "What Do Neural Networks Learn When Trained With Random Labels?",
NeurIPS 2020（arXiv 2006.10455）。原文：

- 「We prove that when trained on random labels, the principal components of weights at the first layer are aligned with the principal
  components of data」（Proposition 1：高斯輸入、各向同性初始化，Σ_w 的特徵空間就是 Σ_x 的特徵空間；證明用對稱性）。
- 「To the best of our knowledge, the alignment effect has not been established in the literature before.」
- 「while these effects are established for training on random labels, we also observe them empirically for real labels」；VGG16、ResNet18、CIFAR10、ImageNet。
- 第一層之後有一節「Deeper layers」。

**這是我們「核的增益對齊輸入主方向」（ρ_align ≈ +0.9）的先行工作**，而 `refs.bib`、正文、補充都沒有引。現稿把對齊歸給 Saxe 2014 的
深線性預測（hostile review 09-15 的 M7），那個歸屬仍然成立，但 Maennel 是「在真實 CNN 上量到並證明第一層」的先例，必須引並劃界：
他們是第一層的權重共變異數對資料共變異數（跨多次訓練或多個濾波器）；我們是每一層的核對它自己的輸入 patch 共變異數、44 個公開權重、
真實標籤、並給出到位的 epoch。**「Deeper layers」那一節要讀完才知道我們在深層的量測有多少是新的**——待做。

## 4. 對證明總帳的更新

| 命題 | 原狀態 | 新狀態 |
|---|---|---|
| Th2 | 可證未證 | **已知**（Shawe-Taylor 2005；Blanchard 2007 含置中與更緊的界；Reiß–Wahl 2020）。論文引用，命題 1 改為引這三篇＋Lawley；我們的是對 k*(τ) 與樣本門檻的應用 |
| Th5／Th5′ | 可證未證 | **已知**（同一批文獻，寬度當樣本數；Koltchinskii–Lounici 2017）。論文把層內律寫成「C 個樣本的樣本共變異數」，引用；數值比較待做 |
| Th8（ReLU） | 可證未證 | **先讀 arXiv 2507.07625 與 Louart–Liao–Couillet 2018 再動手** |
| Th10（權重衰減） | 可證未證 | 先查（Maennel 的對稱性論證、implicit bias 文獻）再動手 |
| Th11（對齊） | 只有實證 | 第一層有 Maennel 2020 的命題；深層待讀其「Deeper layers」節 |

## 5. 做法上的教訓

今天三條命題都是先證後查，三條都查到既有結果，其中兩條連「證不出來」的判斷都是錯的。**以後一律先查再證**：
每條命題動筆前先花一小時找（a）古典矩陣／統計文獻、（b）kernel PCA／隨機矩陣、（c）近兩年的深度學習理論。

## 6. 要加進 refs.bib 的（照慣例先核）

Shawe-Taylor et al. 2005（IEEE TIT）、Blanchard et al. 2007（Machine Learning）、Reiß & Wahl 2020（AoS）、Koltchinskii & Lounici 2017（Bernoulli）、
Maennel et al. 2020（NeurIPS 33）、Cacoullos & Olkin 1965（Biometrika；先讀到全文）。
