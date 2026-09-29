# Th9（殘差流的譜下界）文獻查核（2026-09-29）

對象：`notes/bplus-plan-2026-09-29.md` §4 的命題 Th9——block 的加法輸出 y ＝ x ＋ f（ReLU 前），
(a) rank Σ_f ≤ r ⇒ λ_i(Σ_y) ≥ λ_{i+r}(Σ_x)；(b) Σ_y ⪰ (1 − ρ)(Σ_x ＋ Σ_f)，ρ 是 x 與 f 的最大典型相關。

方法：WebSearch 找候選，**每篇都從 arXiv 抓 PDF、用 pdftotext 抽原文讀定理與相關段落**（WebFetch 不能解析 PDF，照 09-07／09-08 的做法）。
共讀 14 篇。PDF 與文字檔在 session 暫存區，不進 repo。

**結論先講：Th9 的兩條不等式本身是古典矩陣分析，不是新定理；「skip connection 防止秩塌縮」這個機制自 2021 年起已知，而且已知它在
深度趨於無窮時會失效。我們能主張的有三件：(1) 把它寫成通道共變異數的逐特徵值下界，兩個常數（支路的可達秩 r、典型相關 ρ）都量得到；
(2) 在公開的訓練後 CNN 上量這兩個常數與流的實際保留率——沒有找到人做過；(3) 用它解釋兩個預登記實驗（A26 讀滿、A27 待判）。
寫法與 r_max 相同：已知機制，當量測的解釋用。Dong 等人把「有 skip 時的緊下界」列為公開問題，這句可以引。**

---

## 1. 逐篇

| 文獻 | 對象 | 說了什麼（原文） | 對 Th9 的意義 |
|---|---|---|---|
| Dong, Cordonnier & Loukas, ICML 2021（arXiv 2103.03404） | 自注意力網路，任意權重 | 無 skip 與 MLP 時輸出雙指數收斂到秩 1（Thm 2.3）。有 skip 時只有 **Claim 3.1：存在無窮多組參數使殘差不減**（證明是「有一條跳過所有層的路徑」）。原文：「A tight lower bound to the residual in the presence of skip connections is highly nontrivial, and we pose it as an open challenge」 | 機制已知；**一般性的下界被列為公開問題**。Th9(b) 是一個下界，但量的是不同的東西（跨樣本的通道共變異數，不是 token 均勻性的殘差），而且依賴一個量測到的 ρ，不是只由架構決定 |
| Noci et al., NeurIPS 2022（2206.03126） | transformer，初始化 | 「while the claim holds for any finite depth … as L → ∞ a Transformer can still lose rank unless the residual branches are adequately initialized」；支路要 1/√L 縮放 | skip 不是保證。Th9 是逐 block 的，連乘 L 個 block 得 (1−ρ)^L，與這個結果一致——**必須寫明 Th9 不隨深度一致** |
| Daneshmand et al., NeurIPS 2020（2003.01652；已在 refs.bib） | 線性／ReLU MLP，初始化，BN | Lemma 3：無 BN 的線性網路**即使有殘差連接**也收斂到秩一，除非支路隨深度縮小；BN 才給 Ω(√d) 的秩下界（Thm 2） | 同上；而且提醒我們的 ResNet 都有 BN，Th9 的 f 是「最後一個 BN 的輸出」 |
| Feng et al., NeurIPS 2022（2206.06072；已在 refs.bib） | Jacobian 的數值秩 | 附錄 B：skip「shortens the length of the chain of Jacobian matrices, thus restraining rank diminishing」；無定理 | 量的是映射的秩，不是激活共變異數；定性說法已有 |
| Huh et al., TMLR 2023（2103.10427；已在 refs.bib） | 線性網路訓練後的有效秩 | 附錄 I、圖 17：「ResNets, unlike linear networks, do not exhibit a monotonic rank contracting behavior」；「A thorough study on the relationship between residual connections and rank is left for future work」 | **訓練後「殘差不縮秩」的觀察已有**（線性、權重的有效秩）；他們留作未來工作 |
| Li & Papyan, NeurIPS 2023（2401.09018，Residual Alignment） | 訓練後的 ResNet（全連接與卷積），支路的 Jacobian | RA1 表示在一條線上等距前進；RA2 支路 Jacobian 的頂奇異向量彼此對齊；**RA3 全連接 ResNet 的支路 Jacobian 秩至多為類別數 C**；RA4 頂奇異值 ∝ 1/深度；拿掉 skip 就消失 | **與 Th9(a) 互補**：若支路的秩由類別數決定，(a) 說每個 block 最多從流裡拿走約 C 個方向——類別數進入下界。要引，並用量到的 k*(Σ_f) 檢查 |
| Zhang, Jiang et al.（2404.10947v5，2026-05） | MAE／diffusion 的 ViT，訓練後的有效特徵秩 | 恆等捷徑讓深層特徵的秩偏高；把捷徑衰減（α_min 0.5–0.6）得到較低的有效秩與較好的線性探針 | **訓練後「捷徑抬高秩」在 transformer 上有量測**；方向與我們一致，對象不同（ViT、生成式預訓練） |
| Arangath Joseph et al., ICLR 2025（2410.10609，λ-skip） | transformer 與 SSM | Thm 4.1：skip 強度 λ 滿足 λ² − a(S·C_M ＋ \|λ\|)² > 0 時，秩塌縮量度 μ(Y^(K))² ≥ a^K μ(Y^(0))² | 已有「有 skip 時的下界」，但對象是序列模型的 token 均勻性量度，常數來自權重範數；與共變異數的逐特徵值下界不同 |
| Alman & Song（2505.16284） | 自注意力，小權重 | 權重小時即使有 skip 也「layer collapse」（整個網路可被一層近似） | skip 不是充分條件的另一個說法 |
| Cirrincione（2604.23681，2026-04） | transformer | LayerNorm 對仿射秩中性；「residual connections generically obstruct rank collapse in real Transformer architectures such as BERT-Base」 | 定性結論已有（2026） |
| **Everett（2607.14018，2026-07）** | Transformer 前饋 block，**初始化**，Jacobian 與表示的有效秩 | skip 把梯度繞過支路、恢復秩；支路／捷徑比值決定秩塌縮與 ensemble 行為的取捨；擴張寬度服從 Marchenko–Pastur 型的律；初始化的 Jacobian 有效秩預測 CIFAR-10 可訓練性 | **敘事最接近、時間最近**（兩個月前）。差別：初始化對訓練後、Jacobian 對激活共變異數、MLP block 對卷積、無逐 block 不等式。B+ 的脊椎第二段（初始化可算）與它相鄰，**必引並劃界** |
| Haggi-Mani & Rish（2606.10324，2026-06） | MLP 殘差堆疊，訓練後，合成馬可夫鏈 | 訓練後殘差流的有效秩**隨深度單調下降**；§5.1「Measuring the Effective Rank Correctly」：逐位置（flatten）與池化（pool）兩種量法結果不同 | (i) 殘差流訓練後也可能降秩——「殘差保秩」不能寫成全稱；(ii) 他們獨立指出池化與逐位置的差別，與我們的池化結果同型，要引 |
| Barzilai et al., ICLR 2023（2211.14810） | 卷積殘差網路的 GP／NTK 核 | 殘差核的特徵值衰減率與無 skip 的核相近；skip 改善條件數 | 無限寬核的譜，不是有限寬訓練後的共變異數；相鄰 |
| Gadhikar & Burkholz（2210.02411） | 殘差網路初始化的 dynamical isometry | 既有初始化偏向捷徑；提出平衡支路與捷徑的初始化 | 與 §10.15 的 γ＝0／γ＝1 對照相關（初始化時支路開或關） |

## 2. 兩條不等式本身

- **(a)** 是奇異值的和不等式 σ_{i+j−1}(A ＋ B) ≤ σ_i(A) ＋ σ_j(B)（Ky Fan 1951；Horn & Johnson, *Topics in Matrix Analysis*, 1991, Thm 3.3.16）
  取 σ_{r+1}(F) ＝ 0 的直接推論。**定理號憑記憶，未翻書核對**（與 `horn2013matrix` 的 Thm 4.3.28 同樣待查）。
- **(b)** 是 Cauchy–Schwarz 加典型相關（Hotelling 1936）的定義，兩行。
- 所以論文裡兩條都寫成 **Proposition（elementary）**，證明放補充，正文一句話帶過來源；不用「we prove」當賣點。

## 3. 沒有找到的（搜尋過、未見）

- 在訓練後的卷積 ResNet 上，於加法處同時量 Σ_x、Σ_f、交叉項、典型相關，並報告流的逐特徵值保留率。
- 以「支路最後一個 conv 的可達秩」當 bottleneck block 能從流拿走的方向數的上限。
- 把 plain 層（WΣWᵀ，無下界）與殘差 block（有下界）放在同一個分解裡對比。

「沒找到」的信心低於「找到了」：搜尋只用了 WebSearch（美國區結果）、沒有 Semantic Scholar 或 OpenReview 全文檢索；
mechanistic interpretability 社群量過 transformer 殘差流與各元件輸出的餘弦相似度（負值表示抹除資訊），那是逐 token 的向量夾角，
不是共變異數的典型相關，但精神相近，寫作時要提一句。

## 4. 對論文措辭的約束

1. **不得寫**「we show that residual connections prevent rank collapse」。要寫「a known mechanism (Dong 2021; Huh 2023), stated here for the
   channel covariance with two constants that can be measured」。
2. **必須寫明界不隨深度一致**：L 個 block 連乘是 (1−ρ)^L，與 Daneshmand Lemma 3、Noci 的「L → ∞ 仍會塌縮」一致。
3. **不得把「殘差流保秩」寫成全稱**：Haggi-Mani & Rish 在訓練後的 MLP 殘差堆疊上量到降秩。我們的陳述限於量過的網路與任務。
4. Residual Alignment 的 RA3 給了「類別數 → 支路的秩 → 流最多失去多少」這條鏈；`measure_residual.py` 已記錄 k*(Σ_f)，
   可以直接檢查支路的有效秩與類別數的關係。**這是探索性的，量測時才想到，不在 plan §9.21 的預登記內**，寫作時要標示。
5. Everett 2026 要在 Related Work 與 S2（初始化可算）兩處劃界。

## 5. 要加進 refs.bib 的（加之前照慣例核 Crossref／官方 proceedings）

Dong 2021（ICML，PMLR 139）、Noci 2022（NeurIPS 35）、Li & Papyan 2023（NeurIPS 36）、Arangath Joseph 2025（ICLR）、
Zhang & Jiang 2024（arXiv 2404.10947；是否已有正式出版處待查）、Everett 2026（arXiv）、Haggi-Mani & Rish 2026（arXiv）、
Horn & Johnson 1991（Topics）。Alman & Song、Cirrincione、Barzilai、Gadhikar 視篇幅決定。
正文 12 頁的參考文獻空間有限：正文引 Dong、Huh（已有）、Li & Papyan、Everett 四篇，其餘進補充。

## 6. 對證明總帳的更新

Th9 的狀態由「可證未證」改為「**可證（初等）、機制已知；貢獻在量測與兩個可量的常數**」。
要補的仍是：推到 k*(τ)、推過 ReLU、量測（已排隊，plan §9.21）。
