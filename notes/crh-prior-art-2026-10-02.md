# 表徵／權重／梯度三者對齊（CRH／PAH）——B7 與 A22／A32 的先行工作（2026-10-02 早；原文在 `refs/B7b/`）

**為什麼現在查**：P-b7.1′ 判完後第 3 環缺一個關於 M（Σ_out ＝ Δ̃MΔ̃ᵀ）的陳述。動手前先查——想到 Ziyin 等人的 CRH，
repo 裡只有 09-08 survey 表的一行（「表徵／權重／梯度對齊，不是維度剖面」），**原文沒讀過**。這是文獻地圖 L3 的漏洞，今天補。

讀了（PDF 原文）：Ziyin, Chuang, Galanti & Poggio, *Formation of Representations in Neural Networks*（ICLR 2025，arXiv 2410.03006，全文＋附錄 B 證明）；
Ziyin, Wang, Li & Wu, *Parameter Symmetry and Noise Equilibrium of SGD*（NeurIPS 2024，2402.07193，摘要＋§4–5 定理段）；
Tansley, Massart & Cartis, *On the NFA for Deep Neural Networks*（2510.15563，摘要）；Koplow, Poggio & Ziyin, *Ubiquity of Emergent Hebbian Dynamics*（2505.18069，摘要）。
Semantic Scholar 上引 CRH 的只有 15 篇，逐題名看過，沒有做卷積層逐層寬度的。

## 1. 他們做了什麼

**記號（對照我們的）**：一層 h_b ＝ W h_a；H_c ＝ E[h_c h_cᵀ]、G_c ＝ E[g_c g_cᵀ]（g ＝ −∇_h ℓ）、Z_a ＝ WᵀW、Z_b ＝ WWᵀ。
對我們：H_a ＝ Σ_patch、**H_b ＝ Σ_out**、**G_b ＝ Γ**（conv 輸出端的梯度 Gram）、Z_b ＝ WWᵀ、G_a ＝ WᵀΓW ＝ patch-AGOP。

- **CRH（六條對齊）**：前向 H_b ∝ G_b ∝ Z_b，後向 H_a ∝ G_a ∝ Z_a。名稱：RGA（表徵—梯度）、RWA（表徵—權重）、GWA（梯度—權重）。
- **定理 1（噪音—正則平衡）**：線上 SGD＋weight decay γ、全連接層、假設 1（‖h_a‖²、‖g_b‖² 取均場值）。H_b 的一步變化取期望、令其為零：
  0 ＝ z_b(E[g_b h_bᵀ] ＋ E[h_b g_bᵀ]) − 2γH_b ＋ ηz_b² G_b（式 6：學習項、正則項、噪音項）。再加 E[ΔW]＝0（**局部極小**，引理 5：E[g_b h_bᵀ] ＝ γWWᵀ）
  與 E[Δ(WWᵀ)]＝0（引理 6：z_b G_b ＝ γ²WWᵀ）⇒ H_b ∝ G_b ∝ Z_b。
- **定理 2（Master Theorem，純代數）**：一條前向＋一條後向對齊 ⇒ 六個矩陣在某子空間內互為冪次（指數在 [−1, 3]）。表 1 的九個相。
  **相 8：H_a ∝ Z_a 且 G_b ∝ Z_b ⇒ H̃_b ∝ Z̃_b² ∝ G̃_b²**（證明一行：H_b ＝ W H_a Wᵀ ∝ W WᵀW Wᵀ ＝ Z_b²）。這叫 PAH。
- **CRH 何時破**：E[ΔW]＝0 與 E[ΔZ]＝0 不相容（ΔW 的變異不為零）；γ 小時（式 10）前向 H_b—Z_b 與後向 G_a—Z_a 強、其餘弱。
- **實驗**：全連接網（合成資料）、ResNet-18 CIFAR-10 的**倒數兩個全連接層**、ResNet-18 自監督、六層 transformer（各 head 輸出，λ_h ∝ λ_g^α，α ＝ 3／2／1 隨層）。
  卷積層：自監督 ResNet-18 的 conv 層「RGA holds well for the fully connected layers, but not so strongly for the conv layers」，只有共變異數的圖（圖 12–13），沒有數字。
  量法：正規化並去均值後的 Pearson 矩陣對齊 α(A, B)；樣本＝影像（測試集）。
- **自陳的限制**（結論段）：「They apply only to fully connected layers, and a future step is extending them to other types of layers」；只談訓練末段。
- **7.4**：NFA ＝ 後向 GWA 在 B ∝ I 時的特例；「forward GWA」（WWᵀ ∝ 輸出端梯度 Gram）是他們提出的、稱為 forward NFA。
- Ziyin–Wang–Li–Wu 2024：指數對稱（含縮放不變）下 SGD 噪音把參數推到「噪音平衡點」；縮放不變的範數平衡是其特例（與 van Laarhoven 2017、Li & Arora 2020 同一件事）。
- Tansley 2025：深線性網第一層 NFA 的指數是 1/L——冪次關係的指數隨深度變，是已知的。

## 2. 被它蓋掉的（要照實寫）

| 我們的 | 他們的 | 論文怎麼寫 |
|---|---|---|
| **A32 的探索性發現**「核增益首先跟輸出端梯度能量（Spearman +0.98／+0.95）」 | **forward GWA**（Z_b ∝ G_b），2025 年提出並在 FC 層量過 | 寫成「forward GWA（Ziyin et al. 2025）在 conv–BN 層、位置解析的梯度上量到」，不是發現。pending §C 加一條 |
| **A22 的對齊**（核增益對輸入變異，ρ_align +0.91） | **backward RWA**（Z_a ∝ H_a）；NC 文獻在倒數第二層已知，Maennel 2020 第一層 | 已知現象（09-29 已記），這裡多一個名字與理論框架 |
| **P-b7.2**「κ 跟 Γ 的寬度走（+0.86）」 | **RGA**（H_b ∝ G_b）；他們的圖 4：表徵的秩與對齊度負相關 | 寫成 RGA 在 τ 寬度上的讀數；「Γ 定 κ」的方向不新 |
| B7.2′ 的 Σ_out ＝ Δ̃MΔ̃ᵀ | 定理 1 的式 6（H_b 的平衡方程式）是同一件事的穩態版本 | 我們的是「不需穩態、不需局部極小」的精確分解；並列寫 |
| 逐通道範數平衡 | Ziyin 2024、Li & Arora 2020、van Laarhoven 2017 | 已知，引用 |

## 3. 沒被蓋掉的（缺口）

1. **卷積層、位置當樣本**：他們的理論只到全連接層，conv 層只有圖、自陳較弱。我們的 Σ_out 與 Γ 都是位置解析的。
2. **BN 層上定理 1 的局部極小那一步不成立**：引理 5 要 E[g_b h_bᵀ] ＝ γWWᵀ，但逐通道縮放不變給 diag(E[ĝWᵀ]) ＝ 0，而 γ·diag(WWᵀ) > 0——
   這正是 B6（A35 已量：|cos| ≤ 0.04、梯度是 λW 的 1.3–13 倍）。**所以在 conv–BN 層，CRH 的證明路線斷在第二步；留下的只有二階矩的平衡**：
   E[Δ(WWᵀ)]＝0 ⇒ (2γ − ηγ²)WWᵀ ＝ η E[ĝĝᵀ] − (1 − ηγ)(∇L₀Wᵀ ＋ W∇L₀ᵀ)，後一項對角為零（「空心」）。對角就是已知的逐通道範數平衡；
   非對角是他們沒寫的。——這一條是初等的，寫成 remark，不當貢獻。
3. **寬度的後果沒有人寫**：PAH 給「譜是冪次關係」，沒有人把它接到「達 τ 變異的主成分數」。初等引理（§4 的 B8.0）：**τ 寬度對冪次單調不增**——
   指數 α > 1 時表徵比梯度 Gram 窄，窄多少由 α 與 Γ 的譜定。這把 P-b7.1′ 留下的缺口（全寬後段 Γ 475 對 Σ_out 40）變成一個可檢的數：α。
4. **逐層的指數剖面**：他們在 transformer 上看到 α 隨層 3→2→1；CNN 逐層沒有。

## 4. 它給的靈感：第 3 環的陳述（B8；先寫陳述，再登記，再算——plan §9.28）

- **已知（引用）**：Master Theorem 相 8——若後向 RWA（A22 量到的那條）與前向 GWA（A32 量到的那條）成立，則 Σ_out ∝ (WWᵀ)² ∝ Γ²（在共同子空間）。
  兩條若只是單調（s² ∝ λ_a^p、s² ∝ γ^q，共用基底），同一行代數給 λ_b ∝ γ^{q(1＋1/p)}；p＝q＝1 時指數 2。
- **初等（B8.0，不當貢獻）**：λ_1 ≥ … ≥ λ_C ≥ 0、α ≥ β ≥ 0 ⇒ 對每個 k，Σ_{i≤k}λ_i^α/Σλ_i^α ≥ Σ_{i≤k}λ_i^β/Σλ_i^β，故 k*_τ(λ^α) ≤ k*_τ(λ^β)。
  （交叉相乘：i ≤ k < j 時 λ_i^αλ_j^β − λ_i^βλ_j^α ＝ λ_i^βλ_j^β(λ_i^{α−β} − λ_j^{α−β}) ≥ 0。）
- **陳述 B8**：在 conv–BN 層，Σ_out 與 Γ 近似共用特徵基底，Σ_out 沿 Γ 的第 i 個特徵方向的能量 ∝ γ_i^α；於是 **κ ＝ k*_τ(Σ_out) ＝ k*_τ(Γ^α)**，α ≈ 2。
  「κ 跟 Γ 走」（順序）與「Γ 比 Σ_out 寬一個數量級」（P-b7.1′ 的缺口）是同一個冪次關係的兩面；第 2 環的兩條對齊（A22、A32）與第 3 環的寬度是同一個陳述。
- **為什麼可能錯**：(a) 他們自己說 conv 層的 RGA 較弱；(b) 兩條前提我們量到的是秩相關，不是比例；(c) BN 層沒有局部極小，相 8 的代數前提（比例）沒有動力學保證；
  (d) 已看到的數字裡，c10_full 末層 τ＝0.95 時 Σ_out 35 對 Γ 36（像 α＝1），τ＝0.999 時 40 對 250（像 α > 1）——單一指數可能不夠，或頭部平坦所以平方不改 0.95 寬度。
- **用什麼檢**：全部是既有資料——`results/gradwindow/*_epoch_*.npz` 存了每層的 Σ_out 與 Γ **矩陣**（尺寬臂三種子＋λ 掃描的全寬臂，掃描跑完共 6 跑）；
  W 在 `results/wd_sweep/*/` 與 `results/a18_pt/` 的 checkpoint；`results/gradgram/*_spectra.npz` 只有特徵值（21 個 checkpoint，只能做秩配對的版本）。不需要新訓練。

## 4b. 結果（10-02 09:00；plan §10.27）——B8 照登記不成立

預測先提交再跑。**P-b8.1–8.4 全部不成立**：Σ_out 與 Γ_end 在位置解析的 conv–BN 層上不共用特徵基底（共對角比例 0.17–0.23，隨機基底 0.09–0.15）；
指數不是 2（Γ 基底下 0.26–0.48）；逐層中位 Γ 的一次方就在 Σ_out 寬度的 10–17% 內，平方沒有更近；W 基底下 s² ∝ γ^{2–5}，不是相 8 的比例。
**PAH 不延伸到 conv–BN 層**（與他們自陳 conv 層較弱一致）——這本身是一句可寫的劃界：「the forward relations of Ziyin et al. hold on batch-normalised
convolutions in rank only; as matrix proportionalities they fail (co-diagonal fraction 0.2 against 0.1 for a random basis)」。
我在 §2 表裡把 A32 的 +0.98 稱為 forward GWA：**要加限定——秩相關是，冪次不是**（a_ZG 1.8–5.1）。
判後觀察（探索性）：只有塌縮的末三層近似共基底（0.52–0.61 對 0.05），頭部寬度一次方就對、0.999 寬度要平方才對——不是單一冪次；
Σ_out 沿**窗** Gram 方向的能量指數 ≈ 1.0，沿端點 Gram 只有 0.3–0.5（對未量的四跑補登記 P-b8.5／8.6）。

## 5. 要補引並劃界的文獻（加進 `refs.bib` 前照慣例核 Crossref／proceedings）

- Ziyin, Chuang, Galanti & Poggio, ICLR 2025（CRH／PAH）——**必引**：第 2、3 環；劃界句：「stated and tested for fully connected layers; we measure the forward relations on
  batch-normalised convolutions with positions as samples, where the local-minimum step of their proof is unavailable, and draw the consequence for the τ-width」。
- Ziyin, Wang, Li & Wu, NeurIPS 2024（噪音平衡；縮放不變）——與 B6 的三篇 BN 文獻同處引。
- Tansley, Massart & Cartis 2025（NFA 指數 1/L）——冪次指數隨深度變的先例，一句。
- Koplow, Poggio & Ziyin 2025／26（weight decay 使更新呈 Hebbian）——與記憶窗（Wang & Aitchison 2025）同處一句，可不引。
