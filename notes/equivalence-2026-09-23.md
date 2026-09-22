# D3：r_max 下「兩群合併」的效果量與等價檢定（敵意審稿 2026-09-15 M3.3／行動表 D3）

**問題**：§3.2 寫「bounded against unbounded layers differ under C in all 14 models
(Mann–Whitney p < 10⁻²) and in none under r_max (p > 0.01 in all 14)」。後半句用不顯著證明等價，
審稿人 M3.3 直指 absence of evidence。

**做法**（`code/scripts/equivalence_rmax.py`，測試 `tests/test_equivalence_rmax.py`；輸出
`results/equivalence_rmax.txt`、`paper/table_equivalence.tex`，**尚未 `\input`，稿子不動**）：
- 同一批層（1(iii) 的 14 個有 1×1 擴張的訓練 checkpoint，conv 列含 depthwise，n/C ≥ 50），k*(0.999)。
- 效果量：Hodges–Lehmann 位移 Δ = median(b_i − u_j)，與優越機率 A = P(b > u) = U/(n_b n_u)；
  95% CI 由層 bootstrap（兩群各自重抽，4,000 次）。不重抽影像：D1 已量到影像噪音低於 k* 的整數解析度。
- 等價（TOST 式）：Δ 的 90% CI 落在 (−δ, δ) 內。三個邊界：該剖面自己的相鄰層階差（規則 9）、
  0.05（六配方逐層 SD，規則 10）、0.10（上界的一成）。δ_min = max|90% CI 端點| = 能宣告等價的最小邊界；
  它不會小於 CI 半寬，n_b = 20 時約 0.03 是下限。

## 結果

| | 用 C | 用 r_max |
|---|---|---|
| Δ 範圍 | −0.46 … −0.76 | −0.057 … +0.035 |
| A 範圍 | 0.00–0.03 | 0.11–0.67 |
| 95% CI 排除 0 | 14/14 | 2/14（ResNet-34 A1 −0.041、ResNet-50 A3 −0.031） |
| A 的 CI 含 0.5 | 0/14 | 11/14 |

- |Δ_rmax| / |Δ_C| 中位 **0.027**：分母換掉後位移縮到 1/37。
- 等價宣告：自身階差 5/14、0.05 邊界 7/14、0.10 邊界 12/14；δ_min 中位 0.052，最大 0.144（EfficientNet-B0，
  bounded 層譜平、CI 寬；Wide-ResNet-50-2 0.125 同因）。
- **跨模型（14 個 Δ_rmax 各算一個觀測）：中位 −0.018，95% CI [−0.028, −0.004]，11 負 3 正。**
  bounded 層在 r_max 下**系統性地略低於** unbounded 層——「填滿上界」是「到 δ_min 以內」，不是「恰好」。
  與正文已寫的「94% of the bound at ratio four, 99% at six」一致：bounded 到 0.91–0.99，unbounded 到 0.84–0.98。
- 三個 MobileNet／EfficientNet 是正號（bounded 略高，A 0.56–0.67，下界 0.50）；ResNet 家族全是負號。

## 判定

「兩群合併」**降為**「位移縮到 C 版的 3%，且落在跨權重散布（0.05）的量級；殘餘 −0.018 [−0.028, −0.004] 是
bounded 層未完全填滿上界」。p > 0.01 這句不能再用。這**不動搖**分母主張——主張是「文獻忽略了一個已知上界，
用 r_max 當分母後 1×1 擴張層與其他層落在同一水準」，效果量表把「同一水準」量化為 ±0.05 內、殘餘 −0.02。

## 論文改法（稿子解凍時；正文一句＋補充一表 S14）

§3.2 第三個檢定改為：
> *Within a model the two populations merge to within the between-weights scatter*: the Hodges–Lehmann
> shift of the bounded layers against the unbounded ones is $-0.46$ to $-0.76$ under $C$ (probability of
> superiority $\le 0.03$, 14/14 CIs exclude zero) and $-0.057$ to $+0.035$ under $r_{\max}$ (median across
> the 14 models $-0.018$, 95\% CI $[-0.028, -0.004]$; supplement, Table~S14). The residual is a deficit of
> the bounded layers, not a surplus: a trained expansion reaches $94$--$99\%$ of its bound while an unbounded
> layer reaches $94$--$97\%$ of $C$. Equivalence within the six-recipe scatter of $0.05$ holds in 7 of 14
> models and within $0.10$ in 12; the two exceptions (EfficientNet-B0, Wide-ResNet-50-2) have flat bounded
> spectra and wide intervals, not large shifts.

主張表（補充 B）「Normalising by attainable rank removes the bounded/unbounded split」一列的證據欄改為
「Δ shrinks 37× (median ratio 0.027); residual −0.018 [−0.028, −0.004]; max k*(0.999)/r_max ≤ 1 in 22/22」，
去掉 p 值。

補充表 S14 caption：Per-model Hodges–Lehmann shift of bounded against unbounded layers in $k^*(0.999)$ under
$C$ and under $r_{\max}$, 95\% layer-bootstrap CI, probability of superiority $A$, the smallest equivalence
margin $\delta_{\min}$ (largest 90\% CI bound in magnitude), the profile's own adjacent-layer jitter, and
equivalence at the jitter / 0.05 / 0.10 margins. Depthwise layers are counted with the unbounded group, as
in Sec.~3.2.

## 對審稿的回答

M3.3 的「report the median difference with a CI, or a TOST」：兩者都給了。誠實的代價是「merge」要改成
「merge to within 0.05」並承認殘餘 −0.02 顯著；換來的是這一段不再靠 p > 0.01 站著。M3.2「drop the p, report
count and effect size」在 §3.2 同步做掉。
