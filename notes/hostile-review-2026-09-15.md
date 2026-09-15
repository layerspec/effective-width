# 敵意審稿演練 —— 12 頁版（tag v0.1-first-full-draft ＋ commit 617652b）

**2026-09-15。** 對象：`paper/main.tex`（12 頁）＋ `paper/supp.tex`（11 頁）。
扮演：TPAMI 一位「理論嚴密＋統計挑剔＋對量測論文天生懷疑」的審稿人（R2 那種）。
所有引文都對過 `main.tex` 原句、`results/checkpoint_analysis.txt`、
`notes/eb-validation-2026-09-08.md`、三張 `table_*.tex`。

每條給：**位置 → 審稿人會怎麼寫 → 我們能不能擋 → 修法與成本**。
結尾給 major-revision-or-better 的機率估計與依「弱點／成本／增益」排序的行動表。

---

## 0. 審稿人的一段話總評（他會這樣開場）

> The paper audits how three unstated choices (denominator, estimator, tensor) move
> layer-wise effective-width profiles of CNNs, fixes a protocol, and reports a large
> measurement campaign with an untrained control. The measurement work is careful and the
> negative results are reported honestly. My concerns are about *significance*: two of the
> three "pitfalls" are true by construction (a residual sum can exceed the rank of one summand;
> averaging removes within-image variance), the third is a bound already published by Kim et al.
> and Han et al.; the propositions are standard linear algebra; the central causal claim
> ("the reordering belongs to the data or the recipes") rests on an experiment the authors say
> they could not afford; and the one mechanism result (kernel–input alignment) is post hoc with
> the registered sign wrong. The paper reads as a rigorous technical report on a measurement
> protocol. Whether that is a TPAMI contribution depends on the AE.

這段是最危險的：它不攻擊任何一個數字，攻擊的是「所以呢」。下面的 M1–M3 都是在替這段找彈藥。

---

## A. 重大問題（每一條都足以單獨觸發 major revision；M1–M2 若三位審稿人同時抓可能直接 reject）

### M1. 顯著性：三個「陷阱」中兩個是恆真式、一個是已知界

- **位置**：摘要、§1 第二段、貢獻 2、結論。
- **審稿人**：「Pitfall 3（block 輸出超過 conv3 的秩界）is a tautology: rank(A+B) can exceed
  rank(A). Reporting 161/161 is reporting that addition is addition. Pitfall 2（池化較低）：
  global averaging discards within-image variance, so the pooled covariance is a covariance of
  a *different* random variable; that it has fewer effective dimensions is expected. Pitfall 1
  is Han et al.'s bound with a new denominator. The contribution is quantification of three
  expected effects.」
- **擋得住多少**：一半。我們的主張其實是「量級大於文獻報告的效果、且文獻從未講明」——這是
  對的，但論文現在的措辭（"three pitfalls"、"161 of 161"、"687 of 703"）把重心放在計數，
  計數對恆真式沒有說服力。**687/703 那 16 個例外反而是最有資訊的部分**（整數平手），
  現在寫法像在炫耀 p 值。
- **修法**（寫作，0 成本，−0 頁）：
  1. 貢獻 2 改寫為「三個選擇各移動剖面 X 倍於已報告效果」，每個給**倍數**而不是計數：
     分母 4×（擴張比）、估計量 2.2× 自身階差、張量 4.1×。計數退到括號。
  2. 明說 Pitfall 3 是「必然發生、但兩個文獻各讀一邊而不自知」，把新穎性放在
     「Ansuini／E&B 的上升剖面在 conv 輸出處不存在」——這是可反駁的實證主張，恆真式不是。
  3. 摘要第二句 "the disagreement is largely about three choices made without comment"
     ——「largely」沒有量化。§5.1 自己說「would differ in level even if they agreed on both」
     ——即我們**沒有**完全調和兩篇文獻，只是說明各自量了什麼。摘要要與 §5.1 一致：
     "the two profiles are measurements of different objects; on one protocol neither shape
     survives"。

### M2. 中心因果主張「不是 block、是資料或配方」的邏輯不成立

- **位置**：貢獻 3、§3.9 "Is the block the cause?"、§5.2、結論 "owe it to their data or
  recipes, not their block"。
- **審稿人**：「A null result at CIFAR-10 scale with three seeds (T–R 0.50 vs 0.55, p = 0.2)
  does not exclude the block as a cause; it excludes 'block alone at CIFAR-10 scale'. A
  block × dataset interaction is entirely consistent with every number reported: the bottleneck
  network's level *falls* on CIFAR-10 training (0.20 → 0.16), which is itself a block-specific
  behaviour the basic network does not show. The authors acknowledge in §5.3 that the deciding
  experiment 'is the one this study could not afford', yet the conclusion states the negative
  as a finding.」
- **擋得住多少**：不能擋，只能改措辭。這是**邏輯**問題不是資料問題。
- **修法**（兩條路）：
  - (a) 寫作（0 成本）：全文四處改為「the block is not sufficient: under one CIFAR-10 recipe
    the bottleneck network keeps its initial ordering as the basic one does; what reorders the
    ImageNet ResNet-50 checkpoints is therefore not the block alone, and the interaction with
    dataset or recipe is untested」。結論不得再寫 "not their block"。
  - (b) 資料（A17 縮減版，US$120–150、約 3 天 pod）：ImageNet-1k 上 basic-[6,6,6,6] vs
    bottleneck ResNet-50 同配方各 2 種子。這是整篇唯一能把「觀察」升級為「指認原因」
    的實驗，也是審稿人最可能直接開口要的。作者 09-10 決定不跑；**本審查建議重估**——
    若十月投、明年二月收到 major revision 再跑，反而多等半年。

### M3. 統計：無任何抽樣信賴區間；相關層的 Wilcoxon；用不顯著證明「合併」

- **位置**：§3.2（Mann–Whitney 14/14 → 0/14）、§3.4（Wilcoxon p ≤ 1.2e-4）、§3.9（p ≤ 4e-8）、
  全文所有 ρ。
- **審稿人**：
  1. 「Every headline number (687/703, ρ = −0.08, +0.91, 0.84) is a point estimate on one image
     sample. The authors show six *recipes* disagree by SD 0.053, but never resample *images*.
     A block bootstrap over images would cost minutes and is absent.」（門檻清單 A7，未做）
  2. 「Layers are not independent observations (§3.4 admits this in one clause). Wilcoxon over
     53 correlated layers with p = 4e-8 is not a p-value the reader can use; report the count
     and an effect size and drop the p.」
  3. 「'In none under r_max (p > 0.01 in all 14)' uses non-significance as evidence of
     equivalence. Report the median difference with a CI, or a TOST equivalence test.」
  4. 「n = Np with p = 16 positions per image treats positions within an image as independent
     observations for the gate. The robustness check (4 vs 64 positions gives the same profile)
     shows the *estimate* is stable, not that n/C ≥ 50 is the right unit.」
- **擋得住多少**：3、4 可以用現有數字擋（中位數差 0.011；位置消融）。1、2 擋不住。
- **修法**：
  - A7 影像 bootstrap（Mac，半天到一天）：對 ResNet-50 V1 與 VGG-16 的 6,400 張子集做
    200 次影像重抽，給 687/703 類計數、T–R ρ、ρ_align 中位數、pooling drop 的 95% CI。
    進補充一張表 S13，正文每個承重數字後加 CI 或「CI in Table S13」。
  - 正文把三個 Wilcoxon／Mann–Whitney 的 p 值改為「count ＋ median effect ＋ CI」；
    p 值只留補充。這同時解決規則 7–9 的精神。

### M4. 自我違規：E&B 驗證關卡用了 16 個 block，其中 9 個過不了自家 n/C ≥ 50 的門檻

- **位置**：§3.1 第二段 "matches over all 16 blocks at ρ = 0.997"；摘要 "the ImageNet values
  of Elmoznino and Bonner (ρ = 0.997)"；貢獻 1。
- **事實**：block 輸出池化估計量 n = 50,000 張影像；stage 3（C=1024）n/C = 49、stage 4
  （C=2048）n/C = 24。`checkpoint_analysis.txt` §0b 自己寫「gated (pooled n/C >= 50,
  7 blocks): max |dlog| = 0.098」；`notes/eb-validation-2026-09-08.md` 寫「正式版只能寫
  7 個 block；16 個 block 的 r = 0.999 是兩條同樣取樣不足的曲線互相吻合」。
- **審稿人**：「§2.4 says no row with n/C < 50 is reported. §3.1 reports sixteen. The agreement
  at the wide blocks is agreement between two under-sampled estimates and the authors' own
  Proposition 1 says both are biased in the same direction.」（而且 §3.4 又說「the pooled gate
  excludes the 1024- and 2048-channel layers」——同一篇論文兩個標準。）
- **修法**（寫作，0 成本）：§3.1 改為「over the seven blocks that pass our pooled gate the
  median |Δ log| is 0.063 and the maximum 0.098; over all sixteen ρ = 0.997, but at the nine
  wider blocks both estimates fail the gate of §2.4 and agree because they share the bias」。
  摘要的 ρ = 0.997 拿掉或改為「recovers … on every block that passes its own gate」。
  貢獻 1 同步。**這條是零成本、不修必被抓。**

### M5. 摘要把分解結果掛在 44 個 checkpoint 上，實際是 10 個

- **位置**：摘要 "on 44 trained checkpoints of eleven families measured against their own
  initialisations, training moves the bottleneck from the data to the kernel"。
- **事實**：§3.12 「On the 465 dense convolutions of the ten checkpoints in Table S12」、
  「in six of nine checkpoints」、「above the measured value in 401 of 412」。44 是家族表
  （排序）的數量；分解是 10 組。ρ_align 是 48 網路（19 ImageNet ＋ 29 CIFAR）。
- **審稿人**：會當成 overclaim 的證據，連帶懷疑其他數字。
- **修法**（寫作）：摘要拆成兩句：44 → 排序／水準結論；「on ten of them the covariance is
  decomposed」→ 瓶頸從資料到核。

### M6. 理論：命題是教科書等級，且命題 1 的假設在本文自己說的體制下不成立

- **位置**：Prop. 1（閘門）、Prop. 2（截斷一階）、Prop. 3（Ostrowski）、Cor. 1。
- **審稿人**：
  1. 「Proposition 1 relies on Lawley's first-order expansion, whose terms 1/(λ_i − λ_j) blow
     up for near-degenerate eigenvalues, i.e. exactly for the power-law spectra with α < 1 the
     paper says CNN activations have (§1.1). The gap assumption λ_{k+1} ≤ λ_k/2 is never
     satisfied at k* for such spectra. The proposition is thus not the justification for the
     gate; the empirical check on ResNet-50 is.」
  2. 「Proposition 2 is a first-order Taylor expansion. Proposition 3 is Ostrowski's theorem.
     Corollary 1 follows in two lines. These are useful as bookkeeping but should not be
     presented as theoretical contributions.」
- **擋得住多少**：2 擋不住，也不必擋——把「a proposition, a bound and a measurement」（§1）
  的語氣降一級即可。1 要正面回應。
- **修法**：
  - Prop. 1 後加一句：expansion 在 λ_i ≈ λ_j 附近失效；本文以此為**尺度論證**（bias 的
    1/n 標度與 (1−τ) 因子），實際門檻由 ResNet-50 全譜的實算（1.8%／3.3%）定。並把補充的
    「Limiting cases: no spectral gap」段指過去。
  - §1 "each factor comes with a proposition, a bound and a measurement" 改為 "each factor
    is pinned by an identity and a measurement"。
  - 若想留一點理論味：命題 S1（壓縮譜）與 Cor. 1 的雙邊界是唯一非顯然的部分，把它們
    抬到正文、Prop. 2 降到補充。頁數中性。

### M7. 對齊（matched filter）是唯一的機制主張，卻是登記方向錯誤後的 post hoc

- **位置**：貢獻 4、§3.12 "Alignment, measured"、摘要 "becomes a matched filter"。
- **審稿人**：「The registered prediction failed in 0 of 48 networks and the opposite is
  reported post hoc as the paper's mechanism. Honest, but the alignment of a trained linear
  map's gain with its input's principal directions is the textbook behaviour of gradient
  descent on a linear layer (Saxe et al. 2014, which the authors cite), so the finding
  confirms deep-linear theory in a CNN rather than discovering a mechanism. 'Matched filter'
  is also a loaded term: the layer is aligned with the *input covariance*, not with a signal
  template.」
- **修法**（寫作）：
  - 把它定位為「the deep-linear prediction (Saxe) measured in trained CNNs, with its size
    (ρ +0.91) and its time course (epoch 7–15, before the profile settles)」——時間順序是
    真正的新東西，現在埋在段中。
  - 「matched filter」改為「aligned with its input's principal directions」或在第一次出現時
    定義。
  - 登記符號錯誤保留，但補一句為什麼錯（現在有：「amplification of the principal
    directions steepens the output spectrum」，可以）。

### M8. 缺少審稿人一定會點名的引用（砍頁時掉的）

在 `refs.bib` 裡、正文已不引：

| 應引處 | 缺的文獻 | 審稿人會說 |
|---|---|---|
| §3.14 尺定寬重訓 vs 均勻縮 | **Liu et al. 2019, Rethinking the Value of Network Pruning** (`liu2019rethinkingPruning`) | 「The question 'does the per-layer allocation matter or only the parameter budget' is exactly Liu et al.'s, who found uniform budgets match pruned architectures; the authors' P9.7/P9.15 replicate that finding without citing it.」**這是最傷的一條** |
| §1.1 剖面 | **Feng et al. 2022, Rank Diminishing** (`feng2022`) | 深度方向秩單調不增的定理，直接與 Prop. S2 與「隨機剖面隨深度下降」相關 |
| §3.12 核的秩 | **Huh et al. 2023, Low-Rank Simplicity Bias** (`huh2023`) | 訓練後核出現 null 方向（24 層）就是這條 |
| §3.13 診斷 | **Lin et al. 2020, HRank** (`lin2020hrank`) | 用特徵圖秩剪枝的先例，與「尺 vs slimming」段並列 |

- **修法**：四篇各一句，共約 6 行；第 12 頁已滿，需等量刪字（§3.6 或 §3.7 各刪一句即可）。

---

## B. 中等問題（單獨不致 major，合起來影響 AE 對嚴謹度的印象）

- **B1. 同一個量兩個數字**：ResNet-50 T–R 在 §3.2 是 −0.07（全驗證集、一個隨機種子）、
  表 2 與 §3.9 是 −0.08（子集、30 對中位）、§3.9 末段又寫「rather than −0.07」。加一個
  括號說明或統一用表 2 的值。同理 ResNet-50 median k*(0.95)/r_max §3.2 說 0.42（全集）、
  表 2 T 欄 0.47（子集）。
- **B2. Garg 重現的兩個 0.2 差距**：features.27（0.73 vs 0.53）、.30（0.52 vs 0.31）
  「consistent with a recipe their paper does not fully specify」是猜測。13 個點裡 2 個差
  0.2，MAD 0.052 主要靠其他 11 個。要嘛給推測的具體原因（weight decay？epoch？他們用的
  τ 計算方式？），要嘛就寫「unexplained」。
- **B3. τ = 0.95 是登記的主要結果，全文承重卻在 τ = 0.999 與 r_max**：§3 開頭有交代，但
  審稿人會說預登記是裝飾。建議 §3 開頭加一句「the registered primary outcome (k*(0.95)/C)
  is reported for every claim in the supplement claims table; the text leads with the
  statistic the data showed to be the informative one, and says so at each place」。
- **B4. "eleven families spanning five block types"** 但表 2 只有 9 個家族列＋3 個 depthwise
  子列；ConvNeXt-T 在 §3.8 出現但不在表 2（它有 2 個 checkpoint、3 個種子嗎？）。
  需一句話說 ConvNeXt 為何不進家族表（4 個 dense 層，ρ 無意義）。
- **B5. 池化例外的 "two reversals of one and 23 channels"**：23 個通道的反轉不是平手，
  是哪一層？審稿人會問。
- **B6. §3.10 "six of eight checkpoints" 不掉分**：8 = 6 ResNet-50 ＋ R18 ＋ VGG？正文沒列。
  另外 "p ≥ 0.15"（McNemar）用不顯著證明「不掉分」，同 M3.3；給差值的 CI。
- **B7. A18 ImageNet 兩個種子**：表 3 下塊 ±0.43 是兩個種子的 SD，等於兩者之差／√2。
  寫清楚「range」而不是 SD，否則審稿人會說 n=2 的 SD 沒有意義。
- **B8. 補充材料的 ViT 一段**「one checkpoint without an initialisation control」——
  審稿人會問為什麼放。要嘛補一個隨機 ViT（Mac 半小時），要嘛刪。
- **B9. 單一作者、無 ImageNet 受控實驗、A3 未做**——限制段已寫，但 §5.3 的
  "the one this study could not afford" 會被引用成 reject 理由。改為中性：「is left to
  a study with the compute」。

---

## C. 寫作（TPAMI 審稿人常在第一頁決定態度）

- **C1.** §1.1 "The linear count is the object width is about" 讀不通（漏了 "that"？
  → "The linear count is the object that width is about"，或改寫 "It is the linear count
  that width concerns"）。
- **C2.** 摘要 190 字內但**一句 60 字**（"dividing by the attainable rank … 161 of 161
  bottleneck blocks"）；拆成三句。
- **C3.** 全文數字密度過高：§3.2 一段 14 個小數。TPAMI 讀者接受，但每段至少一句
  沒有數字的結論句（"We conclude" 已有，前面的段落沒有）。
- **C4.** "the ruler" 在 §3.13 之前沒有定義就開始用（§3.6 "the ruler and PR"）。第一次出現
  給定義：「k*(τ)/r_max, hereafter the ruler」。
- **C5.** §2.2 兩個 "which width is the point" 式的口語句，TPAMI 可接受，不超過三處。
- **C6.** 結論段落一句 90 字（"the identity … turns the count into … released so that"）。
- **C7.** 圖 1 caption 提 "Generated by scripts/paper_figures.py" ——投稿版拿掉，放
  Reproducibility。表 caption 同。

---

## D. 審稿人會要的補充實驗（依「他要不要得到」排序）

| # | 要求 | 成本 | 沒做的後果 |
|---|---|---|---|
| D1 | 影像 bootstrap CI（M3） | Mac 半天–1 天 | 「no confidence intervals」一句話寫在 summary 裡 |
| D2 | A17 縮減版：ImageNet basic vs bottleneck 同配方（M2） | US$120–150、3 天 | 因果主張只能寫成「not sufficient」，論文高度降一級 |
| D3 | 等價檢定或效果量 CI 取代 14/14 的不顯著（M3.3） | 分析 1 小時 | 「absence of evidence」 |
| D4 | 隨機 ViT 對照（B8） | Mac 半小時 | ViT 段被要求刪除 |
| D5 | ONI（精確等距）一臂（§3.12 末 "was not run"） | pod US$10 | 「the penalties in use meet by half」只對軟懲罰成立，審稿人會問硬約束是否到頂 |

---

## E. 機率估計

以現稿（v0.1）投：**major revision 以上約 45–50%**，低於門檻清單的 60%。
主因不是資料，是 M1／M2／M4／M5 四條——兩條是 overclaim（M4、M5，零成本可修）、
兩條是定位與邏輯（M1、M2）。

修完 A 節的寫作項（M1、M2a、M4、M5、M6 措辭、M7 定位、M8 引用）＋ D1、D3：**約 55–60%**。
再加 D2（A17 縮減版）：**約 65%**——因為它把論文最高的一條主張從「觀察」變成「指認」，
而且是審稿人最會開口要的東西；沒有它，第一輪意見幾乎確定會要求。

**排序後的行動表**（弱點 → 成本 → 增益）：

1. M4 E&B 16→7 block（30 分鐘、必修）
2. M5 摘要 44→10（10 分鐘、必修）
3. M8 四篇引用各一句（1 小時＋等量刪字）
4. M2a 「not sufficient」措辭四處（1 小時）
5. M1 貢獻 2 改倍數、摘要「largely」改（2 小時）
6. C1–C7 寫作（2 小時）
7. B1–B9（半天）
8. D1 影像 bootstrap → 表 S13（1 天）
9. D3 效果量 CI（1 小時，與 D1 同一趟）
10. M6 命題語氣＋Lawley 失效說明（1 小時）
11. M7 對齊定位（1 小時）
12. **D2 A17 縮減版**（作者決定；US$120–150）
13. D4、D5（選配）

1–7 全部是寫作，一到兩天可完成，且不影響 12 頁（M8 需等量刪字）。

---

## F. 處理狀態（2026-09-16）

| 項目 | 狀態 |
|---|---|
| M1 | 做了：貢獻 2 改為倍數（分母 4–6×、池化 2.2× 自身階差、張量＝擴張比）；摘要與結論改「they measure different objects」；§3.5 明說殘差和必然升秩、發現是差距大小與兩個文獻各讀一邊 |
| M2a | 做了：貢獻 3、§3.9 結論、§5.2、結論、限制段五處改「the block alone is not sufficient／alone or in interaction with the block」；§3.9 補 bottleneck 水準下降 vs basic 上升（0.21→0.32） |
| M2b／D2 | 未做，作者決定 |
| M3／D1／D3 | D3 部分做了（§3.2 給 r_max 下兩群中位數差 ≤ 0.056 的效果量）；影像 bootstrap 未做 |
| M4 | 做了：§3.1 改為 7 個過門檻 block（中位 \|Δlog\| 0.035、最大 0.098、ρ 0.96），16 block 的 0.997 標為「both estimates carry the bias」；摘要與貢獻 1 同步；補充 survey 表 E&B 的張量由 conv out 改 block out |
| M5 | 做了：摘要拆成 44 個排序／水準、10 個分解 |
| M6 | 做了：§I 改「identity and a measurement」；Prop. 1 後加展開在特徵值聚集處失效、門檻由實算譜定 |
| M7 | 做了：對齊定位為 Saxe 深線性預測的量測（大小＋時間順序）；「matched filter」全文刪除 |
| M8 | 做了：Feng 2022（§1.1）、Huh 2023（§3.12）、HRank（§3.13）、Liu 2019（§3.14）各一句 |
| B1 | 做了（−0.07 全集單種子／−0.08 子集 30 對；0.42 V1 全集／0.47 子集） |
| B2 | 做了（「unexplained, recipe only partly specified」） |
| B3 | 做了（§3 開頭一句） |
| B4 | 做了（表 2 caption 說明 ConvNeXt-T 為何不在） |
| B5 | 做了（DenseNet-121 一通道、tv2 ResNet-50 `layer4.2.conv1` 23 通道） |
| B6 | 做了（列出八個 checkpoint、3,200 張、差值先於 p） |
| B7 | 做了（caption：兩種子 SD，兩全寬種子差 0.60） |
| B8／D4 | 隨機 ViT 對照已量（`results/vit/*_random_layers.csv`），見補充 ViT 段 |
| B9 | 做了（「left to a study with the compute」） |
| C1–C7 | 做了（C5 未動；C7 caption 的 Generated by 全拿掉，改在 Reproducibility 一句） |
| 頁數 | 加字後 13 頁；等量刪字：Robustness、Threshold 兩小節移補充；bib 去 11 個 arXiv note；fig1／fig9 0.92 欄寬；§2.1 尾句、§3.6 數字、§3.12 Prop. S1/S2 敘述、§5.1 首段、失敗預測清單壓縮 → 12 頁 |

外部審稿（作者另請的兩個 LLM，`review/*.docx`，09-15）與本文重疊處已隨上表處理；未處理的是 R2 的「標題把 covariance 寬度說成 effective width」（改標題由作者定）與 R2 要求的介入實驗（超出預算）。
