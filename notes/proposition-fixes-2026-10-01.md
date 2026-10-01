# 命題的嚴謹度修正（2026-10-01 下午；審稿 agent #15–20、#45–47 的回應；**正文等框架審稿過閘再貼**，這裡是改好的陳述與證明）

## 1. 命題 1（取樣偏差）——比值的 O(1/n) 項與高斯假設（#16、#15）

**現況**：主文把 E F̂(k) − F(k) 寫成 Lawley 項的等式「to first order in 1/n」，但 F̂ ＝ S_k/T̂ 是比值，T̂ ＝ tr Σ̂ 的變異也是 O(1/n)；高斯假設沒寫；n ＝ Np 被稱為 conservative。

**改後的陳述**：設樣本為 n 個 i.i.d. 高斯向量（Lawley 1956 的假設），總體特徵值 λ₁ > … > λ_C 兩兩相異。則
  E F̂(k) − F(k) ＝ L_k ＋ D_k ＋ O(n⁻²)，
  L_k ＝ (1/(n tr Σ)) Σ_{i≤k} Σ_{j>k} λ_iλ_j/(λ_i − λ_j) ≥ 0（Lawley），
  D_k ＝ (2/(n (tr Σ)²)) [ F(k) Σ_{j>k} λ_j² − (1−F(k)) Σ_{i≤k} λ_i² ] ≤ 0（delta method：−Cov(S_k, T̂)/t² ＋ Var(T̂)·E S_k/t³，Var T̂ ＝ 2 tr Σ²/n，Cov(S_k, T̂) ＝ 2Σ_{i≤k}λ_i²/n ＋ O(n⁻²)）。
D_k ≤ 0 的證明：Σ_{i≤k}λ_i²/Σ_{i≤k}λ_i ≥ Σ_{j>k}λ_j²/Σ_{j>k}λ_j（頭部的均方大），乘開即得。上界：若 λ_{k+1} ≤ λ_k/2 則 L_k ≤ (2k/n)(1−F(k))，而 D_k ≤ 0，所以 **E F̂(k) − F(k) ≤ (2k/n)(1−F(k)) ≤ (2C/n)(1−F(k))** 照舊成立；「偏差非負」只在 L_k ≥ |D_k| 時成立（實測譜上成立，見表）。

**數值核對**（`scripts/prop1_delta_check.py`，高斯樣本、置中樣本共變異數、n ＝ 50C、1500 次）：

| 譜 | k | E F̂ − F（模擬 ± se） | L_k | D_k | D/L |
|---|---|---|---|---|---|
| 冪律 α＝1.5 | 4 | 4.9e-4 ± 0.9e-4 | 4.6e-4 | −3.7e-5 | −0.08 |
| 冪律 α＝1.5 | 32 | 7.1e-4 ± 0.1e-4 | 6.9e-4 | −5.8e-6 | −0.01 |
| 冪律 α＝1.0 | 4 | 7.8e-4 ± 1.0e-4 | 9.5e-4 | −2.0e-5 | −0.02 |
| 兩倍譜隙 k＝8 | 8 | 3.1e-4 ± 0.4e-4 | 3.7e-4 | −6.7e-6 | −0.02 |

讀法：D_k 是 L_k 的 1–8%，方向為負；α＝1.0（近重根）時 Lawley 本身高估 ~20%（二階項），這正是主文已寫的「expansion loses its content where eigenvalues cluster」。**改法**：陳述加 D_k 與高斯假設；「to first order」改成上式；n ＝ Np 那句改為「Np 高估獨立樣本數（位置相依），門檻的實際依據是 Fig. S1 的經驗平坦化」。

## 2. 命題 S3（砍通道的一階效應）——期望值界與 greedy（#18）

Deshpande–Rademacher–Vempala–Wang 2006 的界是**期望值**：E_S‖A − π_S A‖²_F ≤ (m＋1)‖A − A_m‖²_F（volume sampling）。改寫：「the right inequality holds in expectation over a volume-sampled S」。
Greedy：Altschuler et al.（ICML 2016, PMLR 48:2539–2548）定理 1 是雙判準界——greedy 取 r ＝ 16k/(ε σ_min(OPT_k)²) 個欄位達到 (1−ε) 倍最優，帶 σ_min 因子；**沒有 (m＋1) 型的 Frobenius 界**。改寫：「greedy selection, which the controller uses, carries only a bicriteria guarantee with a conditioning factor (Altschuler et al. 2016, Thm. 1); it is a heuristic here」。bib 加 `altschuler2016greedy`（PMLR 卷頁已核）。

## 3. 命題 S5（注水最佳性）——陳述與證明對上（#17）

現況：D_ℓ 在 shrink 側定義為 CSSP 殘差 tr R_S/tr Σ_a，證明卻用 D_ℓ(C) − D_ℓ(C＋1) ＝ λ_{C+1}/tr Σ（特徵值尾）。CSSP 殘差對 |S| 不保證凸，greedy 的 R_S 也不是尾。
改法：命題只對**譜替身** D̃_ℓ(C) ＝ Σ_{i>C} λ_i(Σ_a)/tr Σ_a（shrink）與未滿足變異（grow）陳述——D̃_ℓ 凸且差分遞減，注水是 KKT 解；正文一句「the controller optimises the spectral surrogate; the column-subset residual it realises is bracketed by Proposition S3」。ω_ℓ 要定義；「greedy discrete exchange terminates」改為「the discrete water-filling is the greedy solution of a separable convex program, hence optimal」。

## 4. 命題 S6（層內律）——降為 remark、列初始化（#19）

第一句（給定 Σ_patch，WΣWᵀ 與 W̃ΣW̃ᵀ 同律）確實是 Wishart 的定義，不是結果；第二句是 Silverstein 1995。改為 **Remark**：「for Gaussian W the output covariance is a scaled Wishart matrix with population covariance Σ_patch and C degrees of freedom; its τ-width is therefore a functional of the spectrum of Σ_patch alone (Silverstein 1995)」。535 層的檢查改稱「the measured k* of every untrained layer falls inside the Wishart law's distribution (median error ≤ 2 channels, 94–100% within 0.05 r_max)」——它檢的是取樣位置的相依與 BN 沒有把律破壞，不是律本身。
旋轉不變性的前提（高斯）逐家族核過（10-01，torchvision 0.2x／timm 1.0 原始碼）：ResNet／VGG／DenseNet／MobileNetV2／V3／EfficientNet（torchvision）＝ `kaiming_normal_`；timm ResNet／VGG／DenseNet ＝ `kaiming_normal_`；timm EfficientNet／MobileNetV3 ＝ `normal_(0, √(2/fan_out))`（`_init_weight_goog`）；ConvNeXt／ViT ＝ `trunc_normal_(std 0.02)`（截斷在 ±2，即 ±100σ，實質高斯）。**所有量測家族的 conv 都是高斯初始化**；PyTorch 裸 `nn.Conv2d` 的 kaiming-uniform 不在其中。這段進補充一句。

## 5. 命題 S7（整條剖面只透過第一層 Gram）——改成它真正的內容（#20）

內容＝第一層的旋轉不變性：高斯 W₁ 使 X₁W₁ᵀ 的律只透過 X₁X₁ᵀ 依賴影像；後面各層是前一層激活的確定性函數。證明中「BN acts sample-wise」錯（訓練模式用批次統計量），改為「BN is a function of the whole activation matrix, so the conclusion is unchanged」。命題改名「first-layer rotation invariance」並降為 remark，或刪。

## 6. S2、S4、Cor. 1 的小修（#45–47）

- S2：「each almost surely in the range of Σ_act」→「each centred vector a.s. in range(Σ_act)」（ReLU 後均值非零；結論對置中向量成立，秩的計數加 1 給均值方向）。
- S4：「the variance a wider site can admit」的界只對 Σ∘（κ＝1）成立；改寫成方向的陳述：「a site of width C passes at most the top C directions of its input (any kernel); the variance bound is for the isometric counterfactual」。
- Cor. 1 末句：κ＝1、C_out ≥ d 時 min(k*, r_max) ＝ k*，刪「truncated to r_max」。

## 7. 其餘（#36、#57）
- 「the bound is never exceeded」移到 Methods 當管線檢查。
- III-C 的「2–50 倍」段移到承認對象差異之後。
