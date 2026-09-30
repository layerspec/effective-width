# 地圖 L1：初始化時的譜與秩（2026-09-30 開始；讀原文；兩欄產出）

**問題**：隨機權重的網路，每層輸出的（通道）共變異數譜由什麼決定？我們量到的是：隨機 VGG／ResNet 的 k*(0.95)/r_max 剖面隨深度下降
（後段 0.05–0.16），順序跨種子穩定（R–R 0.9+），水準與順序都隨初始化方式改變（γ＝0／γ＝1、LayerScale、JAX）；層內律 S6（Silverstein）
在 11 個家族 535 層上命中；ReLU 步的閉式不合（相關 patch）。這條線要回答：這些現象有哪些已被理論描述、用的是什麼對象與假設。

## 1. 逐篇（讀了的部分；PDF 在 `refs/L1/`、`refs/th8/`）

### Poole, Lahiri, Raghu, Sohl-Dickstein & Ganguli, NeurIPS 2016（arXiv 1606.05340）；Schoenholz, Gilmer, Ganguli & Sohl-Dickstein, ICLR 2017（1611.01232）
- **對象**：全連接、寬度→∞、i.i.d. 高斯權重與偏置；追蹤單一輸入的長度 q^l 與兩個輸入的相關係數 c^l。
- **結果**：長度圖與 C-map 的不動點 (q*, c*)；(σ_w, σ_b) 平面分成有序相（c*＝1：所有輸入的表徵趨於同一方向）與混沌相（c*<1）；
  臨界線上訊號傳得最深；深度尺度 ξ_c 決定可訓練深度（Schoenholz）。
- **與我們的關係**：有序相的 c*→1 就是「表徵共變異數只剩一個方向」——**隨機剖面的水準隨深度下降有理論**：每過一層，樣本間相關往 c* 收，
  共變異數的頭部特徵值吃掉更多變異。我們量的是通道共變異數（跨影像與位置），他們算的是樣本間相關；兩者由同一個 Gram 矩陣連著。
  **沒有的**：有限寬度、BN、卷積、譜的形狀（他們只有一個相關係數）。

### Xiao, Bahri, Sohl-Dickstein, Schoenholz & Pennington, ICML 2018（1806.05393）
- **對象**：隨機 CNN、通道數→∞；共變異數 Σ^l_{α,α'} 是**同一張影像不同空間位置之間**的（也可跨影像）；層對層的 C-map 含空間卷積。
- **結果**：不動點 Σ*（q*、c*）；偏差在不動點附近的線性化由 DFT 對角化——**不同空間頻率以不同速率衰減**；Delta-Orthogonal 初始化讓
  一萬層 vanilla CNN 可訓。
- **與我們的關係**：**這就是相鄰位置 patch 相關的理論**——Fan–Wang 假設樣本近似正交，Xiao 明寫位置之間的共變異數與它的傳遞。
  我們 09-27 的 ReLU 閉式失敗，理論上該用 Xiao 的位置共變異數遞推，而不是 Fan–Wang 的獨立樣本遞推。
  **沒有的**：BN、殘差、有限通道數、譜（同樣只有 q、c）。

### Daneshmand, Kohler, Bach, Hofmann & Lucchi, NeurIPS 2020（2003.01652）
- **對象**：隨機初始化的深層**線性**網路（實驗含 ReLU）、有無 BN；表徵 H_l ∈ R^{d×N}（d 寬度、N 樣本數）。
- **量**：soft rank rank_τ(H)＝#{σ_i²/N ≥ τ}（門檻計數，與 Masarczyk 的數值秩同型）；r(H)＝Tr(M)²/‖M‖²_F，M＝HHᵀ/N——**就是 participation ratio**；
  引理 1：rank_τ ≥ (1−τ)² r(H)。
- **結果**：無 BN 的線性網路（含殘差！）秩塌到 1（隨機矩陣乘積的譜不穩定）；**有 BN 則 PR 沿深度平均 ≥ Ω(√d)**（定理 2／14，Markov 鏈工具），
  對每一層也成立（推論 15）；ReLU 網路實驗同型。
- **與我們的關係**：**「初始化水準」的第一個理論下界**。我們隨機 BN 網路後段的 k*(0.95)/C 是 0.05–0.16；√512/512＝0.044、√64/64＝0.125——
  同一個量級。可以直接檢驗：我們存的隨機網路譜（`local6400/seed*`、`bplus_controls/gamma1`、A30 的隨機 ViT）算 PR，對 √C 畫圖。
  這是「實驗驢證理論」該有的形式。**沒有的**：譜的形狀、卷積、非線性的證明、初始化方式的角色。
  **注意**：他們的殘差結果（無 BN 的殘差網路也塌到秩一）與我們 γ＝1 深 ResNet 初始水準低（ResNet-101 0.08）方向一致。

### Fan & Wang 2020；Louart, Liao & Couillet 2018（見 `th8-prior-art-2026-09-30.md`）
- ReLU 步的譜律在「樣本近似正交」下是 MP 映射∘（線性縫合＋白噪聲）；X 任意時有確定性等價。相關 patch 違反假設。

### Dong, Cordonnier & Loukas, ICML 2021（2103.03404）；Noci et al., NeurIPS 2022（2206.03126）；Giorlandino & Goldt, ICLR 2026（2505.24333）
- 待讀原文。已知結論：純注意力的 token 表徵秩隨深度雙指數塌縮；skip 與 MLP 阻止；訊號傳播理論給初始化的可訓練圖。**對象是序列內 token 的秩**，
  與我們的通道共變異數不同（A30 的 attention proj 在初始化只剩 1–3% 的維度，是這個現象在通道側的樣子）。

## 2. 待讀（這條線走到前沿）

- Yang, Pennington, Rao, Sohl-Dickstein & Schoenholz, ICLR 2019「A mean field theory of batch normalization」——BN 網路的訊號傳播與梯度爆炸。
- Hanin & Rolnick 2018／Hanin & Nica 2020——有限寬度修正（我們的網路寬度 64–2048，不是無限）。
- Pennington, Schoenholz & Ganguli 2017（動力等距）；Saxe 2014（深線性的學習動力，L3 也用）。
- Everett 2026（transformer 前饋 block 的秩保持）；Martens et al. 2021（deep kernel shaping）；Roberts–Yaida–Hanin 2021 全書的第 4–5 章（有限寬度的四點函數）。
- 卷積特有的：Novak et al. 2019（CNN-GP）、Garriga-Alonso 2019——無限通道極限下卷積網的核，含位置結構。

## 3. 這條線回答了什麼、留下什麼（暫定，讀完前沿再改）

- **已回答**：隨機深網的表徵為什麼隨深度失去維度（有序相、隨機矩陣乘積）；BN 為什麼擋住（PR ≥ Ω(√d)）；位置之間的相關如何傳遞（Xiao）；
  ReLU 步在獨立樣本下的譜律（Fan–Wang）。
- **留下**：(a) 有限寬度、卷積、BN、ReLU 同時在場時，**譜的形狀**（不只一個相關係數或一個下界）——我們的層內律 S6 加量到的 ReLU 步是這個洞的經驗版；
  (b) 初始化方式（γ、LayerScale、權重尺度）如何改變剖面的順序——文獻談可訓練性，不談順序；(c) 相關樣本下的非線性 RMT（Cranston 2026 剛開始碰）。

## 4. 給我們的（第二欄）

1. **驗證 Daneshmand 的下界**：現有隨機網路的譜算 PR 對 √C——不是新實驗，是既有資料對既有定理。
2. **用 Xiao 的位置共變異數遞推重試 ReLU 步**，取代 Fan–Wang 的獨立樣本版本（若要繼續 S2 的「可算」）。
3. 論文脊椎第二段的寫法：「初始化的剖面由架構決定」在文獻裡對應「隨機網路的訊號傳播由 (架構, 初始化) 決定」；要引 Poole／Schoenholz／Xiao／Daneshmand，
   並說我們加的是譜的形狀與逐層的量。
