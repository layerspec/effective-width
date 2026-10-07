# 目標 (c) 的先行工作：ReLU 步的 Hermite／Hadamard 冪展開（2026-10-07 18:00 讀；PDF 在 `refs/B13/`，不進 git）

| 論文 | 做了什麼（讀到的） | 給我們的 | 行動 |
|---|---|---|---|
| **Joudaki & Hofmann, "Emergence of globally attracting fixed points in deep neural networks with nonlinear activations", arXiv 2410.20107 v2（2024-10；讀摘要、Prop. 1、Lemma 1、Cor. 1、Thm 1、Prop. 2–3）** | 均場（d→∞）下兩個輸入的核 ρ_ℓ 由核映射 κ 決定性演化；Mehler 引理＋Hermite 展開給 κ 的閉式 κ(ρ)＝Σ_k c_k² ρ^k；Thm 1：非線性激活下 ρ_ℓ 全域收斂到唯一不動點（0 或正）；**Prop. 2：殘差強度 r 的核映射 κ_ψ ＝ r²κ＋(1−r²)·id，不動點不變、收斂隨 r 變慢**；Prop. 3：正規化層只縮放核映射。 | 我們 stage 內的流遞推的**純量均場版**（已在草稿 III-A 引 Prop. 2）。他們的對象是兩個輸入的相似度，不是通道共變異數的 τ 寬度；均場、全連接。Hermite 展開本身（Mehler 1866）是工具，不是他們的。 | 草稿 III-A 已引；目標 (c) 的 Σ_z 公式引 Mehler／此文 Lemma 1 |
| **Murray, Jin, Bowman & Montúfar, "Characterizing the spectrum of the NTK via a power series expansion", ICLR 2023（arXiv 2211.07844 v4；讀 §4 Thm 4.1／4.3、Obs. 4.2／4.4）** | 核 Gram nK ＝ Σ_p c_p (XXᵀ)^{∘p}（Hadamard 冪，係數由激活的 Hermite 係數與深度定）；**Thm 4.3：置中核的有效秩 eff(K̃) ≤ eff(XXᵀ)·Σ_{p≥1}c_p/c_1**（ReLU 淺層比值 ≈ 1.3）——輸入 Gram 低秩則置中核低秩。假設：樣本單位範數、資料高斯（NNGP）。 | **直接相關**：ReLU 輸出的共變異數 Σ_z（C×C）與樣本核 K（n×n）非零譜相同，所以 Thm 4.3 說 ReLU 步最多把有效秩放大常數倍（≈1.3）。我們量到：初始化 bottleneck ≈ 0、basic −15%、訓練後 +50%——都在常數倍內。τ 寬度不是 eff rank，但方向一致。 | 目標 (c) 的理論上界引它；我們補的是 τ 寬度版本與量到的失效 |
| Daniely, Frostig & Singer, NeurIPS 2016（已知，未重讀）；Cho & Saul 2009（arc-cosine）；Pennington & Worah 2017、Fan & Wang 2020、Louart 2018（`th8-prior-art` 已查） | 對偶激活／Hermite 展開的核；單層 CK 的極限譜（高斯資料） | feasibility C 已用 arc-cosine 與高斯等價在 VGG 單層上試過，差 0.05–0.2——**單層的位置樣本不高斯**。目標 (c) 換對象：殘差流是幾十個分支的和，高斯假設可能站得住。 | — |

**結論**：ReLU 步的閉式（給定 Σ_y、高斯假設）是已知的（Mehler／Daniely／Joudaki–Hofmann），它對秩的控制也是已知的（Murray 2023 Thm 4.3）。
目標 (c) 能做的是：在殘差流上檢驗這個已知閉式對 τ 寬度的預測（三種行為），報告失效程度；若命中，整條 stage 遞推（層內律 → 自由卷積 → Hermite ReLU 步）成為**可預測的已知映射鏈**，不是新定理。
