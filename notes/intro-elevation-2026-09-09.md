# §I 拔高立意草稿（2026-09-09；A18／對齊結果後定稿）

原則：每一句大話後面都要有稿子裡的一個表或一個命題。不得寫出資料還沒撐住的話。

## 開場三段（英文草稿）

**Paragraph 1 — the bottleneck.**
Of the two quantities that define a convolutional architecture, depth has a theory and width has only limits.
Depth has expressivity results and a notion of effective depth; width has the infinite-width limit, in which nothing is learned, and minimum-width theorems, which say when a network *can* represent a function and nothing about how much of its width a trained network *uses*. In between sits the question every width decision actually turns on, and it has no ruler with a guarantee: outside the lazy regime there is no a-priori estimate of a trained layer's effective width, and the a-posteriori measurements that exist contradict one another on the same architectures.

**Paragraph 2 — why the field could not settle it.**
The contradiction is not a disagreement about networks. It is that each measurement makes three choices in silence, the denominator, the covariance estimator and the tensor read, and each choice moves a profile by more than the effects the papers report (Sections V-B to V-E: 14/14, 687/703, 161/161). A literature that cannot fix its instrument cannot accumulate, and the theoretical side offers no anchor because its results hold only where features are not learned.

**Paragraph 3 — what this paper does about it.**
Before its nonlinearity a convolution's output covariance is exactly W Σ_patch Wᵀ. That identity is the paper's fulcrum: it splits the effective width of a layer into what the architecture permits (the attainable rank r_max), what the data supplies (Σ_patch) and what learning does (the alignment of W with Σ_patch), and each factor comes with a proposition, a bound and a measurement. On that footing the paper (i) fixes the instrument and validates it against two published profiles, (ii) derives the sample-size gate and the working threshold rather than choosing them, (iii) measures 22 checkpoints of eleven families against their own initialisations under a pre-registered plan, including the negative results, and (iv) shows what the ruler is for: a diagnostic of where width is unused, a width prescription tested by retraining [A18], and a mechanism for why training compresses [P9.10/P9.11]. [Items in brackets are conditional on tomorrow's results; drop or invert per the pre-registered consequences.]

## 需要撐住的證據（每句對應）
- 「寬度只有極限」→ §II-E（minimum width、lazy）已有。
- 「事後量測互相矛盾」→ Garg vs E&B，§I 現有。
- 「三個選擇各移動剖面」→ Table pooling、Fig blockout、§V-B。
- 「推導閘門與閾值」→ 命題（閘門）§IV、命題（截斷）§V-K、A19。
- 「預登記含否定」→ §V.J、§V-L、附錄 B。
- 「診斷」→ §V-M；「處方」→ A18（待）；「機制」→ 命題 4＋P9.10（待）。

## 符號表（B13）要收的符號
k*(τ)、k*/C、k*/r_max、r_max、C、d、τ、n/C 閘門、Σ_out／Σ_patch／Σ_∘、W／W_∘／P、κ、τ'／τ''、ρ_align／ρ_align_sigma、R–R／T–T／T–R／T–R_stage、ρ_depth、PR／erank／stable rank、S_lin／S_act。
掃描腳本：`scripts/notation_check.py`（抓首次出現未定義、同一量的兩種寫法如 k^*(0.95)/C 與 k*/C）。
