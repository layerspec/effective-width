# Th8（ReLU 這一步的譜律）：先行工作查核（2026-09-30）

**問的問題**：給定一層的輸入共變異數譜，隨機權重＋ReLU 之後的輸出共變異數譜是什麼？這是 B+ 脊椎第二段（初始化剖面可算）
從「層內律」（命題 S6，線性步，Silverstein）推到「整條剖面」還缺的一步。09-27 的 feasibility 筆記 C 節試了三個閉式
（arc-cosine、高斯等價、只有線性項），都與量測不合；當時已引 Pennington–Worah 2017、Louart–Liao–Couillet 2018、Fan–Wang 2020，
但沒有讀它們的**假設**。這次讀了（PDF 在 `refs/th8/`，不進 git）。

## 1. 已知的結果

| 文獻 | 結果 | 假設（與我們對象的關係） |
|---|---|---|
| Louart, Liao & Couillet, *Ann. Appl. Probab.* 28 (2018) 1190–1248 | Gram 矩陣 G＝Σᵀσ(WX)ᵀσ(WX)／T 的 resolvent 有確定性等價（定理 1）；**X 是任意確定性矩陣**，σ Lipschitz，W 高斯（或 Lipschitz 變換的高斯） | 單層；X 任意——這是唯一不要求樣本近似正交的結果；等價式用到 X 的 Gram 與 σ 的一個常數，數值上要解不動點 |
| Fan & Wang, NeurIPS 2020（arXiv 2005.11879） | 多層 CK 譜 μ_ℓ＝MP_γℓ((1−b_σ²)＋b_σ²·μ_{ℓ−1})（定理 3.4）：每一層是「前一層譜的線性縫合＋白噪聲」再過一次 Marchenko–Pastur 映射 | **假設 3.2(c)：樣本近似成對正交**（獨立樣本、d₀ ≫；(ε_n, B)-orthonormal），σ 二次可微、bσ、aσ 由高斯計算；線性寬度極限 |
| Pennington & Worah 2017；Benigni & Péché 2021；Péché 2019 | 單層 CK 的極限譜（高斯資料） | 資料高斯 i.i.d. |
| Adamczak 2025（arXiv 2507.07625） | 非線性隨機矩陣的集中不等式；CK 的線性譜統計量的估計，「（可能相依的）隨機矩陣」 | 集中，不是極限律；相依指的是矩陣之間 |
| Schröder, Cui, Dmitriev & Loureiro 2023（arXiv 2302.00401） | 深層隨機特徵的樣本共變異數之跡的確定性等價；測試誤差的高斯普適性 | 高斯輸入 |
| Cranston, Wang, Kemp & Mahoney 2026（arXiv 2605.29669） | 線性等價在非線性可分資料（XOR）上失效的機制：二次等價、尖峰出現 | 說明線性等價有邊界 |

## 2. 對我們的意義

1. **Th8 不是新的**：ReLU 步的譜律在「樣本近似正交」下已由 Fan–Wang 給出閉式遞推，在「X 任意」下由 Louart–Liao–Couillet 給出
   確定性等價（要解不動點）。我們 09-27 測的「高斯等價／只有線性項」就是 Fan–Wang 那條律的兩個版本。
2. **它們失敗的原因文獻已經寫在假設裡**：我們的樣本是同一張影像的相鄰位置 patch，彼此高度相關，違反 Fan–Wang 3.2(c)。
   09-27 的診斷（「高斯 copula＋真實邊際在 τ＝0.999 全部進 0.03；邊際不是高斯」）與這個解釋一致：位置相關讓 patch 能量的變異數
   大於高斯情形，尾端的邊際就偏離高斯。
3. **還能做的**：(a) 用 Louart–Liao–Couillet 的定理 1（X 任意）解不動點，看它在我們的相關 patch 上準不準——這是他們的定理，
   我們只是驗證適用範圍，寫成一段就好；(b) 不要再試自己證一條新的律：相依樣本下的非線性 RMT 是另一篇數學論文（09-27 已寫過同樣的結論）。
4. **寫法**：脊椎第二段的「可算」只到層內（S6）與經驗的層間傳遞；ReLU 步引 Louart–Liao–Couillet 與 Fan–Wang，
   並說明它們的假設在卷積 patch 上不成立、我們的 copula 診斷量到了差多少。這是誠實的定位，不是貢獻。

## 3. 給我們的靈感

- Fan–Wang 的遞推是「MP 映射 ∘（線性縫合＋噪聲）」；我們的層內律 S6 是同一個 MP 映射（Silverstein）。**若把我們量到的 ReLU 步
  （copula＋邊際）當黑盒子接在 S6 之後，就有一條可算的整條剖面**——經驗的一步夾在兩個定理之間。這比 09-27 的「層間仍實證」更具體。
- Cranston 2026 指出線性等價在有結構的資料上會漏掉尖峰；我們的 patch 譜是冪律、頂端孤立特徵值多（Silverstein 密度法失準的原因，
  feasibility B 節）——同一件事。

## 4. 要加進 refs.bib 的（照慣例先核）

Louart, Liao & Couillet 2018（Ann. Appl. Probab.）、Fan & Wang 2020（NeurIPS 33）已在 feasibility 筆記引到，查 bib 是否已有；
Adamczak 2025、Schröder 2023、Cranston 2026 視篇幅（補充）。
