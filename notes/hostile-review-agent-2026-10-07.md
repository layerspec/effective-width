# Hostile review of the framework (unanchored agent, 2026-10-07)

Scope: `notes/framework-review-package-2026-10-02.md` (as of commit 06001ad) and `notes/theory-chain-draft-2026-09-30.md` (same commit), checked against `results/checkpoint_analysis.txt`, `notes/analysis-plan.md` §9/§10, `notes/preregistration-ledger-2026-09-25.md`, `notes/proofs-2026-09-29.md`, `notes/theory-chain-2026-09-30.md`, `paper/main.tex`, `paper/supp.tex`, and the per-run CSVs under `results/` (a17, a29b, a31). No `hostile-review*`, `radar*`, `framework-review-own*` or `review/` file was opened. Where a number below is "computed here", it was computed from the stored `epoch_*_layers.csv` files with the paper's own gate (dense conv, n/C ≥ 50, k*(0.95)/r_max), not by rerunning any measurement.

Severity: FATAL = would on its own make me recommend rejection of the framework as stated; MAJOR = must be fixed before the claim can stand; MINOR = wording, bookkeeping, or a weak but non-load-bearing point.

---

## (a) Findings

### A. The central empirical statement (3-D, "the order of initialisation is lost at depth on ImageNet")

**1. FATAL — the "reorder" is a property of the statistic, not of the networks: it disappears within stages.**
Claim: "ImageNet ResNet-50 and a 52-layer basic network under one recipe T–R −0.25/−0.26 and −0.21/−0.02 … the order is lost" (package 3-D; draft III-D).
Objection: the all-layer Spearman ranks layers of different stages against each other, so a level shift between stages changes the rank correlation without any within-stage reordering. The primary record already contains the within-stage numbers and they say the opposite: public ResNet-50, six recipes, within-stage T–R +0.48 / +0.55 / +0.57 for layer1–3 and +0.12 for layer4 only (`results/checkpoint_analysis.txt` §6, "Within-stage" table); pooled-rank within-stage T–R for the public families is +0.27 (ResNet-50), +0.75 (ResNet-18), +0.55 (ResNet-34), +0.32 (ResNet-101), +0.57 (Wide-ResNet-50-2), +0.66 (ResNeXt-50) (§8 "within-stage (pooled ranks)" lines). Computed here from `results/a17/resnet50_s{0,1}/epoch_{000,090}_layers.csv`: all-layer −0.25/−0.26, but layer1 +0.86/+0.87, layer2 +0.15/+0.43, layer3 −0.40/−0.15, layer4 +0.30/−0.15. The "reorder" lives in stage 3 (and partly 4) of one architecture; stage 1 keeps the order as strongly as any CIFAR network.
Fix: report within-stage and within-role T–R everywhere, drop every all-layer T–R from load-bearing sentences, and restate 3-D as "the late stages of deep ImageNet networks reorder; the early stages do not."

**2. FATAL — the same public ResNet-50 keeps the order under the nominal denominator.**
Claim: 3-D attributes the reorder to "the data at this depth".
Objection: `checkpoint_analysis.txt` §6: ResNet-50 trained-vs-random ρ is **+0.55 [+0.53, +0.64] under k*/C** and **−0.07 [−0.13, +0.10] under k*/r_max**; §8: "resnet50 under k*/C: T–R +0.57". The sign of the headline phenomenon flips with the normaliser the paper introduces in M1. Inside the bottleneck, by role, the reorder is carried by the 1×1 layers (conv3 −0.51, conv1 −0.28, conv2 0.00; §6 "By conv role"), i.e. exactly the layers whose r_max correction is a factor 4. A referee will read 3-D as an interaction between the r_max normalisation and the 1×1 expansions, not as a property of ImageNet. Nothing in the package addresses this.
Fix: show the reorder with a normaliser-free statistic (e.g. within-role, same-C comparisons; or the raw k* within a role), or state that the reorder is defined only under r_max.

**3. FATAL — the statistic that carries 3-D flips sign with an arbitrary choice of reference draw, and the package says so.**
Claim: "what decides T–R is the reference draw, not fidelity to the weights' true starting point (A33: P9.68/69 hold)" (package 3-A); the same ResNet-50 weights read "−0.25 against γ = 1 and +0.41 against γ = 0" (3-D).
Objection: a quantity whose sign is set by which untrained network one chooses to compare with cannot support "the order of initialisation survives / is lost". ResNet-34 goes +0.61 → +0.13, ResNet-101 +0.20 → −0.59, Wide-ResNet-50-2 +0.33 → −0.24 between the two references (plan §10.16 B). "Every T–R names its reference" is bookkeeping, not a fix: the title candidate "Initialisation Sets the Order" and the 3-D statement have no reference-independent content.
Fix: either define the order statistic against the network's own initialisation only (then public families cannot be used at all), or replace T–R by a reference-free statistic (T–T across recipes vs R–R across draws, or a profile-shape statistic), or drop the order narrative.

**4. FATAL — the A31 ten-class cell ("ten classes at 224 px do not restore the order", P9.81 fails, P-b9.7 "held") is computed on a degenerate profile.**
Claim: "Ten classes at 224 px do not restore the order: same-seed T–R +0.03 / +0.10 (P9.81 fails, two seeds)" (package 3-D, 3-C B9 paragraph; draft III-D).
Objection: computed here from `results/a31/k10_s{0,1}/epoch_925_layers.csv`: the final median k*(0.95)/r_max is **0.021 (seed 0) and 0.008 (seed 1)**; the most common final k*(0.95) is **1 channel in 15/53 layers (seed 0) and 29/53 layers (seed 1)**; plan §10.29 itself records "全網中位 0.21→0.02／0.01" and T–T between the two seeds only +0.65. A Spearman correlation over 53 layers more than half of which are tied at one channel measures tie-breaking noise. The cell was trained for 925 epochs on 10,000 images (fixed 37,000 steps by design), with best top-1 at epoch 639–686 and final top-1 2.0–2.4 points below best (`result.json`): a grossly over-trained network. "T–R is a rank statistic, not affected by level" (plan §10.14) is false at the floor. The registered consequence "class count is excluded" and the B9 "prediction that held against the registered opposite" both rest on this cell.
Fix: re-measure K10 at matched epochs (not matched steps) or at best-epoch checkpoints, report the tie fraction, and withdraw P9.81/P-b9.7 as evidence either way. The same check is needed for K20/K50 (median final level 0.12–0.14; `k*` of 3–8 channels common).

**5. MAJOR — zone thresholds (reorder ≤ +0.10, keep ≥ +0.40) are smaller than the seed spread inside a cell, with two seeds per cell and no confidence interval anywhere.**
Claim: "K = 20 and 50 reorder as well (−0.35/−0.17, +0.03/−0.52)" (package 3-D); A17 basic "−0.21 / −0.02"; A28 K100-64 "−0.11 / +0.16".
Objection: K50's two seeds differ by 0.55, basic6666's by 0.19 with within-stage signs reversed between seeds (computed here: s0 layer3 −0.22, s1 layer3 +0.72; s0 layer4 −0.37, s1 layer4 +0.44). Under the null, Spearman over 53 independent values has SE ≈ 0.14; layers are autocorrelated along depth, so the effective SE is larger. "Zone" assignments at ±0.1 with n = 2 are noise. P9.44 is counted as "holds" with an effect (0.08, 0.19) smaller than the seed spread (plan §10.23) — the package's own rule 9 (effect next to jitter) is not applied to T–R.
Fix: ≥ 3 seeds per cell, a permutation or block-bootstrap CI for every T–R, and no categorical zone language.

**6. MAJOR — attribution by exclusion ends where the design cannot go, and the remaining candidate is confounded with the stem.**
Claim: "Attribution stops at 'the data at this depth, and optimisation length'" (3-D).
Objection: every CIFAR arm uses a CIFAR stem (3×3 stride 1, no max-pool; `code/scripts/trajectory_cifar.py` lines 69–100), every ImageNet arm an ImageNet stem; the spatial grid per stage, the number of positions P entering the paper's own cap P·K−1, and the patch statistics all change together with "the dataset". A28's 64-px arm keeps the ImageNet stem (2×2 last stage) and so is not the CIFAR layout. "The data at this depth" is therefore "data × stem × positions × steps" with no cell that separates them. A referee will ask for CIFAR-10 at 224 px through the ImageNet stem and ImageNet-100 at 32 px through the CIFAR stem (one of each already exists in spirit: A7 upsampled CIFAR to 224 for the random profile only).
Fix: run the two crossing cells or stop the attribution at "ImageNet-style training".

**7. MAJOR — "initial profiles fall with depth" (the premise of 3-D) is not true across families.**
Claim: draft III-D: "At initialisation the profile falls with depth … ρ(depth, k*) at epoch 0 is −0.46 to −0.91 in the five trajectory families".
Objection: draft III-A(iii) says, of ten public families, "ρ(depth, k*₀.₉₅/r_max) between −0.3 and +0.8 … none monotone"; `checkpoint_analysis.txt` §6 gives ConvNeXt-T random ρ(depth) **+0.86 / +0.80**. The assembled consequence in III-D needs a falling initial profile; the inverted-residual and ConvNeXt families do not have one, and the package then explains their T–R by the same mechanism anyway.
Fix: state the premise per family and restrict III-D to the families where it holds.

**8. MAJOR — mixed reference draws inside one ranking sentence.**
Claim: draft III-D (ii): "Within one role (library-default references): 3×3 layers keep the order in … ResNet-18 +0.64, ResNet-34 +0.55 … and lose it in ResNet-50 +0.09, ResNet-101 −0.08 …".
Objection: "library default" is torchvision γ = 1 for ResNet-50 and timm γ = 0 for the other families (plan §10.16 (ii), §10.22). Under one reference (γ = 1) ResNet-34's all-layer T–R is +0.13 (§10.16 B); the within-role values under γ = 1 are not given at all. The sentence violates the package's own correction (i).
Fix: recompute within-role T–R under both references for every family and print both.

**9. MAJOR — the depth axis (A29b) is read from all-layer T–R and is reference-dependent.**
Claim: "the turn lies between 20 and 36 convolutions (P9.50 holds)".
Objection: computed here, ResNet-34 within-stage T–R: layer1 +0.60/+0.90, layer2 +0.20/+0.79, layer3 −0.76/−0.44, layer4 +0.09/+0.14 — the "turn" is stage 3 of ResNet-34 reordering, not "36 convolutions"; under γ = 0 the public ResNet-34 reads +0.61 (§10.16). `checkpoint_analysis.txt` §30 still lists one seed per architecture with "T-T +nan" while the package reports two seeds: the primary output has not been regenerated.
Fix: as in 1 and 3; regenerate §30.

**10. MAJOR — "class count enters monotonically" (3-C) is contradicted by the authors' own ImageNet dose–response, which the same section reports as failed.**
Claim: 3-C: "Class count enters monotonically: the ruler keeps 0.88/0.92/0.98/0.99 … (3/3 seeds)"; 3-D: "the response is not monotone in K (P9.80, P9.84 fail)".
Objection: the monotone statement is the post-ReLU τ = 0.999 ruler on CIFAR-100 subsets at 32 px (A24); the order statement is the conv-output τ = 0.95 T–R on ImageNet (A31). The package uses the first to supply κ(K) for the chain and then reports that the chain's order prediction has no K dependence. Both the direction (Harun 2024; Masarczyk 2023; Garg 2024's CIFAR-10 vs CIFAR-100 parameter ratio) and the mechanism (fewer classes → lower-rank late features) are published; what is new is a number that the ImageNet experiment then fails to connect to the order.
Fix: keep A24 as a measured number on CIFAR; remove "monotone" from the chain's statement of κ.

### B. Pre-registration and circularity

**11. FATAL — B9's "predictions committed before the spectra were opened" were derived from the same spectra, and the lemma was rewritten after the first judgement.**
Claim: "Predictions committed before the spectra were opened: in trained ten-class VGG-16 on CIFAR-10 the spectral cliff sits at 39/40 = 4 positions × 10 classes in 18/18 layers" (3-C B9); "the one quantitative statement that holds".
Objection: plan §9.29 opens with "來源：B8 的逐層表裡 wd5e-4_s0 末三層 k*₀.₉₉₉(Σ_out) ＝ 40／40／40" and states "K ＝ 10 的寬度已見" and lists A18's ruler widths of 40 as already known. P-b9.1 and P-b9.3 are predictions about K = 10 arms whose widths had already been observed and from which the P·K hypothesis was formed; only P-b9.2 (K = 13, 20) is blind, and it is judged on 9 of 18 layers after the pre-stated ratio ≥ 3 filter removes the other 9 (plan §10.28). P-b9.4 (within-cell variance ≤ 0.02) failed on the first checkpoint and the lemma's premise was changed from w_cp to w_out in a "補記 (07:26)" after seeing that checkpoint's log (plan §9.29). P-b9.9 is marked non-blind by the authors. The whole sequence — B8 registered 07:12, judged 07:16; B9 registered 07:22, judged 07:23; part-B thresholds added 07:24, judged 07:41 — ran on data already on disk.
Fix: present B9 as an exploratory observation on VGG-16/CIFAR-10 at 32 px (K = 10, 13, 20), confirmed blind only at K = 13/20 on the layers with a cliff; do not call it a prediction.

**12. MAJOR — Lemma B9.0 is the definition of k*(τ), not a result.**
Claim: "Lemma (elementary: total-variance decomposition + Ky Fan): if the variance outside the span of the (class × position) cell means is ≤ 1 − τ, then k*_τ ≤ P·K − 1".
Objection: "if at least τ of the variance lies in a subspace of dimension PK − 1 then at most PK − 1 principal directions are needed to capture τ" is the definition of k*(τ) applied to a subspace of known dimension; the only content is the identification of the subspace, which is measured, not derived. The lemma gives no condition for when w_out falls below 1 − τ, and the package admits it fails at 48 px (P-b9.10/9.11), on ImageNet ResNet-18 (P-b9.8), and at K = 50/100. One architecture, one dataset, one resolution.
Fix: state it as an observation ("where the cliff exists it sits at P·K − 1 and the cell means are independent"), with the Ky Fan sentence as a remark.

**13. MAJOR — P-b8.5/8.6 and P-b10.1 are registered after seeing most of the data and counted as pre-registered holds.**
Claim: "0.94–1.01 on four runs unseen at registration, P-b8.5" (3-C); ledger: P-b8.5 "看過兩跑全寬＋尺寬臂後補登記"; P-b8.3 "聚合寬度已見，非盲"; P-b10.1 "非盲，α≈1 的後果".
Objection: the "four unseen runs" are the remaining seeds/arms of the same sweep, measured with the same code; the hypothesis was fitted on 5 of 9 runs and confirmed on the other 4 of the same design. This is a split-sample check, not a pre-registered prediction, and it is listed among the 77 "holds".
Fix: label such items "post hoc, confirmed on held-out runs of the same sweep" in Table S17 and exclude them from the hold/fail tally.

**14. MAJOR — the ledger tally is unstable and includes vacuous holds.**
Claim: package §2 (06001ad): "164 statements: 77 hold, 56 fail …"; the previous commit of the same file: "162 … 61 hold, 47 fail"; my count of the ledger's verdict column: 65 hold / 53 fail.
Objection: three different tallies within a week for the same object. "Holds" include P-b7.1′ (admitted vacuous: both widths at C), P9.44 (effect < seed spread), P9.29 (holds by letter, "does not test the allocation"), P-b9.1/9.3 (non-blind, finding 11). With 164 tests and no multiplicity control, ≈ 8 would pass at α = 0.05 by chance; the paper never says what the base rate of its thresholds is.
Fix: one script, one count, a column "blind / non-blind / post hoc", and a sentence on multiplicity.

**15. MINOR — `results/checkpoint_analysis.txt` §31 contradicts the package.**
§31 reads "k20 … (no finished runs) … P9.80 … not judged yet; P9.83 not judged yet; P9.84 not judged yet" while the package and ledger report them judged. The primary analysis output is stale; the K20/K50 numbers exist only in plan §10.29 prose.
Fix: regenerate.

### C. Theory

**16. FATAL (for the theory axis) — the question's second half is unanswered by the authors' own account.**
Claim: the question is "how much of it can be computed before training and predicted during it" (§1).
Objection: §7.1: "κ has no derivation; B8 … failed"; 3-C: "B7 is an inclusion and an ordering, not a bound"; "This locates κ in a measurable object; it does not derive it" (B10); B9 "is a conditional cap with one set of matching measurements". The after-training half of the question has no theory; the before-training half is a Monte-Carlo simulation with Gaussian surrogate images (feasibility note A) plus Silverstein for the linear step. A TPAMI reader of a "theory section" will find one exact proposition (B6) that restates scale invariance, one lemma (B7.1) that extends a known EMA identity to momentum, one identity (B10) and one definition (B9.0).
Fix: stop calling III a theory section; call it "what the published theorems say about this object and where their hypotheses fail", and move the exact statements to a supplement.

**17. MAJOR — Proposition B6 is a straw man by the package's own 10-07 edit.**
Claim: "Theorem 3.5 is stated for fully connected networks without normalisation (their §2); in a conv–BN layer its stationarity equation has no non-zero solution, so it does not transfer" (package 3-C, 06001ad); map-L2 line 85: "Zangrando 2024 全文無 BN".
Objection: showing that a theorem stated for unnormalised networks does not apply to normalised ones is not a result; the mechanism (⟨∇L₀, W⟩ = 0 under BN) is the classical scale invariance (van Laarhoven 2017; Li & Arora 2020 Lemma 1.3), cited by the draft itself. The "verification" |cos| ≤ 0.043 on 21 × 13 layers measures an identity (it should be 0 up to ε and float error); the "stationarity residual 1.6–13" shows SGD has not converged, which is expected at finite learning rate with batch noise and says nothing about rank. Corollary B6′ ("a stationary direction requires ∇L₀ = 0") is a two-line consequence that any reader supplies.
Fix: a one-sentence remark with the four citations; no proposition.

**18. MAJOR — the 3-0 "one principle" is vacuous.**
Claim: "Every width bound in the paper is this principle [Ky Fan] on a particular covariance".
Objection: k*(τ) is defined through partial sums of eigenvalues, so any statement about k*(τ) is "Ky Fan on a particular covariance". The spine names a definition and calls it a principle. It unifies nothing that was not already unified by the definition of the object.
Fix: delete 3-0's first bullet or make it one sentence of notation.

**19. MAJOR — "three widths" mislabels orthogonalisation as the PCA optimum and rests on a sub-noise accuracy difference.**
Claim: "the width a PCA-optimal (orthogonal) encoder would use — forcing exact kernel isometry raises the width from 0.35–0.36 to 0.68–0.73 r_max and costs 0.10–0.16 points (D5)".
Objection: (i) an isometric kernel (WᵀW = I) reproduces the spectrum of the input patch covariance (Ostrowski with θ = 1); that is "the width of the input", not a reconstruction-optimal encoder — PCA's optimum is a projection onto the top-k eigenvectors, which does *not* preserve the full input spectrum. (ii) The registered criterion P-d5.3 was "accuracy within ±0.5 of the none arm" and it *held* (ledger): 93.41/93.84/93.58 vs 93.83/93.74/93.74 (one seed up, two down; `checkpoint_analysis.txt` §13). A mean difference of −0.16 with seed SD ≈ 0.2–0.3 is zero. The spine's claim that the PCA-optimal width "costs" accuracy is built on a number the authors' own registration judged as no change.
Fix: drop "PCA-optimal"; say "isometric"; drop the accuracy cost.

**20. MAJOR — "a Markov chain of covariances" contradicts III-A(ii).**
Claim: 3-0: "At initialisation Σ₁ → Σ₂ → … is a Markov chain of covariances with the first-layer patch Gram as its only data input".
Objection: III-A(ii) states the ReLU step at τ = 0.999 "needs the full per-channel marginals", i.e. Σ_{l+1} is not a function of Σ_l. The chain is Markov in the law of the full activation tensor, not in covariances. `prop:gram` (supp.tex 453) is first-layer rotation invariance — the authors' own `proposition-fixes` §5 says it should be "降為 remark，或刪".
Fix: say "Markov in the activation law; the covariance recursion is closed only for the linear step".

**21. MAJOR — K-FAC / Fisher / active-subspace attributions are loose.**
Claim: "Σ_patch and Γ are the two K-FAC factors of the layer's Fisher (Grosse & Martens 2016)"; "the gradient-side count on Γ — which is Cui et al.'s … 'number of active neurons'".
Objection: K-FAC's backward factor is the Gram of pre-activation gradients under *sampled* labels (the Fisher); Γ here is the empirical gradient Gram with true labels (the empirical Fisher) — a known and non-trivial distinction (Kunstner, Hennig & Balles 2019). Cui et al.'s count is on flattened activations with a cut-off; reading it "at channel level over positions" is a different estimator. Neither mapping is wrong as motivation, but both are presented as identities.
Fix: "analogous to", with the empirical-Fisher caveat.

**22. MAJOR — P-b7.2 ("κ follows Γ", Spearman +0.86) is a K confound.**
Claim: "k*(Σ_out) follows k*(Γ) over 21 checkpoints (Spearman +0.86)".
Objection: the 21 checkpoints are K ∈ {10, 13, 20, 50, 100} × 3 seeds plus step arms; both widths rise with K (A24, P9.79: last-layer k* vs K +0.96). No partial correlation controlling K was registered or reported for P-b7.2 (only for P-b7.6, where it falls to +0.58). Within one network (fixed K) the only evidence is the within-network ordering of P-b7.1′, which the package admits is a weak test "both at C".
Fix: report Spearman within K, or partial on K.

**23. MAJOR — B7's inclusion is vacuous and its bound unmeasurable; the λ "dose–response" is a step function.**
Claim: "Σ_out ⊆ span(Γ_win) (B7.1) … two-sided τ-width bound with constant cond(M)" (3-0, 3-C).
Objection: the inclusion k*(Σ_out) ≤ k*(Γ_win) + 1 is satisfied with Γ_win at 475 against Σ_out at 40 (3-C); M is "不可由終態量測" (proofs note B7.2′), so cond(M) is never measured; B7.2′ is a statement about a quantity that does not appear in the data. The "late level falls with λ" is 0.42 → 0.078 → 0.076 → 0.070: one step between "no decay" and "decay", then flat within 12 % — plan §10.32 says so and cites Fan 2025's invariance. The draft's sentence "the late level falls with λ among arms of equal accuracy" overstates.
Fix: write "weight decay must be present; above 5e-4 the level is insensitive".

**24. MAJOR — B10 is an identity and two of its four registered predictions failed, yet 3-0 lists it as a chain link.**
Claim: "Σ_out = Γ^{1/2} M̃ Γ^{1/2} (B10)" among "one chain, two directions".
Objection: plan §9.31: "這段是恆等式加定義，不是定理"; P-b10.2 fails (14/18), P-b10.3 fails (16/24), P-b10.1 is non-blind; the stem rank bound rank M̃ ≤ rank Σ_patch is r_max in different notation. An identity true for any two PSD matrices with nested ranges is not a link in a causal chain.
Fix: B10 as a descriptive decomposition table in the supplement.

**25. MAJOR — internal contradiction in III-C: the position-sampled covariance "does not collapse" and "collapse holds per position".**
Claim: draft III-C: "NC1 is reached in the pooled representation, and neither the weights nor the position-sampled covariance collapse"; same section B9: 99.9 % of the position-sampled variance lies in a 39-dimensional cell-mean subspace.
Objection: a covariance with 99.9 % of its variance in a (PK − 1)-dimensional subspace *is* collapsed; the "2–50 K" headline is a factor P = 4 times K − 1 (and for ImageNet a factor 49). The "measured degree of hypothesis failure" of collapse theory is then not a failure of the theorems but a comparison with an object (positions as samples) they never addressed — which III-C also concedes. One cannot both count it as hypothesis failure and admit it is a different object.
Fix: decide: either the object differs (then the collapse theorems are not "tested"), or report the P-factor as the measured relation.

**26. MAJOR — the ONI "counterfactual that hits" is close to tautological.**
Claim: "forcing exact kernel isometry moves the profile by the predicted amount (|out − pred| 0.021–0.031)".
Objection: the prediction is the Ostrowski identity with θ = 1: Σ_out has the spectrum of Σ_patch when WᵀW = I. Enforcing WᵀW = I exactly (erank 1.000) makes the identity hold by construction; the only empirical content is that training under ONI leaves the input-patch width of each layer at the none-arm value. That is a much weaker statement than "the decomposition predicts the effect of orthogonalisation", and it says nothing about SO/SRIP realising "half" except that those kernels are not isometric.
Fix: state it as "the input-side width is unchanged by ONI", not as a counterfactual prediction.

**27. MINOR — B7.1 "exact for SGD with momentum" is an exercise.**
The EMA identity of Wang & Aitchison is exact for SGD; adding momentum is solving a linear recursion; the non-negativity condition is a sufficient condition met with a 10× margin by any realistic recipe. Fine as a lemma in a supplement, not as "one exact lemma made sharper than the known EMA view" in the contributions.

**28. MINOR — Γ "≥ 3K wide everywhere" is weak (30 of 512 channels), and the EL2N forward formula over-counts by 1.6–2.1× (P-b7.5 fails) while the text keeps "reproduces the order exactly".**

**29. MINOR — Prop. 1 (prop:gate) as it stands uses Lawley's expansion without the Gaussian assumption and without the ratio term; the authors' own `proposition-fixes` note lists Prop. 1, S3, S5, S6, S7 as needing correction. None of this is in the manuscript yet (package §7.8).**

### D. Known results framed as the authors'

**30. MAJOR — M1 (r_max) is Garg 2024's denominator applied to a different literature.**
Claim: contribution 2 of `paper/main.tex` still reads "The nominal denominator understates a 1×1 expansion layer …" as a finding; the package concedes "the bound is known (Han 2021; Kim 2018) and is Garg 2024's denominator".
Fix: one sentence: "we apply the attainable-rank normalisation of Garg et al. to the activation-spectrum literature, where it was not used".

**31. MAJOR — the nearest-neighbour gap is overstated.**
Claim: §6: "none … asks what sets the layer-wise profile or computes it before training."
Objection: Harun et al. 2024 explicitly ask which factors (class count, resolution, width, depth, augmentation) set the layer-wise tunnel; Masarczyk 2023 (tunnel), Recanatesi 2019 (compression/expansion per layer), Ansuini 2019, Feng 2022 (rank diminishing), Huh 2023 ask what shapes the profile; mean-field work (Poole 2016; Xiao 2018; Yang 2019) computes the initial covariance recursion; Garg 2024 gives per-layer utilized rank. Not cited in the package and to be checked: Cohen, Chung, Lee & Sompolinsky 2020 (per-layer manifold geometry), Doimo et al. 2020 (hierarchical nucleation), Stephenson et al. 2021 (per-layer geometry along training, late-layer memorisation), Parker et al. 2023 / Galanti et al. 2022 (intermediate-layer collapse), Kunstner et al. 2019 (empirical Fisher, finding 21), Andriushchenko et al. 2023 (EMA view of weight decay).
Fix: the gap sentence must be "no one measures the τ-width of the position-sampled conv-output covariance against a random draw across families" — true, and much smaller.

**32. MINOR — "variance ≠ task importance is also the finding of weighted-SVD compression (Hsu 2022)" imports a weight-SVD result on transformers as support for an activation-covariance statement on CNNs.**

**33. MINOR — the "two reproduction gates" are lenient.**
Garg: two of 13 layers differ by 0.20 (0.73 vs 0.53; 0.52 vs 0.31; `main.tex` 456–460), i.e. 40 % relative error on the layers where the profile moves most; r = 0.981 is carried by the 0.07–1.0 range. E&B: `main.tex` 476–477 says "agreement between two implementations of one estimator rather than a reproduction". Yu & Wu: fc1 within 0.009 but qkv level 0.746× (A30), so the gate passes for one role.

### E. Statistics and protocol

**34. MAJOR — paired tests over layers treat layers as independent.**
Claim: §2: "Statistics are paired over all gated layers (Wilcoxon / sign test …)"; "(M4) six recipes, all p ≤ 3·10⁻⁸".
Objection: adjacent layers of one network are strongly autocorrelated (the package's own "adjacent-layer jitter" is a measure of it); Wilcoxon p-values over 53 layers assume exchangeable pairs. The counts (687/703) are also over 22 checkpoints that share architectures. The p-values are decorative; the effect sizes are what matter, and for M2 they are reported.
Fix: report effect sizes with block-bootstrap CIs over images (already done, D1) and over seeds; drop p-values that assume independent layers.

**35. MAJOR — n = 1 trained checkpoint for 9 of 11 CNN families.**
Claim: 3-D family-level statements ("3×3 order kept in VGG-16/ResNet-18/34 and lost in every deep ImageNet family").
Objection: VGG-16 "two checkpoints are with and without BN, so its T–T compares two architectures" (`checkpoint_analysis.txt` §8 Reading); DenseNet-121 has 2 checkpoints and within-3×3 T–T 0.68; the ConvNeXt dense statement rests on 4 layers (p = 0.013). Layer-level and role-level detail outside ResNet-50 has no error bar, as §7.4 admits, yet the family table is the paper's Table I.
Fix: as §7.4 says; or restrict family-level claims to ResNet-50 and the controlled trainings.

**36. MAJOR — the sample-independence hole undercuts the paper's own gate.**
Claim: §2: "a layer is reported only if n/C ≥ 50" with n = 6,400 × 16; III-A(iii): positions of one image are "the duplicated samples that Joudaki et al. explicitly exclude".
Objection: the effective sample size is between 6,400 and 102,400; `proposition-fixes` §1 concedes "Np 高估獨立樣本數 … 門檻的實際依據是 Fig. S1 的經驗平坦化". The gate's justification is empirical, not Prop. 1, and for C = 2048 layers the effective n/C may be far below 50, biasing k̂* downward (the direction of Th2) exactly in the late ImageNet layers where 3-D claims a level rise.
Fix: estimate the effective n per layer (e.g. from the block bootstrap) and gate on it, or show the profile is flat in the number of positions per image at the deep wide layers.

**37. MINOR — ConvNeXt "trained ≠ random only for dense layers (p = 0.013)" on 4 layers is not evidence; nor is depthwise p = 0.21 on 18.**

**38. MINOR — the pooling claim's exceptions are explained post hoc as "integer ties at C = 32"; fine, but the DenseNet drop/jitter ratio of 0.5 is reported as "18/19 models" in the package without the ratio.**

### F. Internal inconsistencies between package, draft, plan and results

**39. MAJOR — "optimisation steps, not image count" (3-C) contradicts A24/A25.**
Claim: "the contraction deepens with optimisation steps, not with image count".
Objection: P9.34 was reversed: at equal epochs 6,400 images give a *shallower* contraction than 50,000 (d 0.74 vs 0.52; `checkpoint_analysis.txt` §25); image count has an effect of the opposite sign; only at fixed steps does the step count dominate (P9.35). "Not with image count" is wrong as written.

**40. MAJOR — the title candidate preferred by the author is contradicted by the package.**
"Initialisation Sets the Order, the Task Sets the Level": 3-D says the order is lost in every deep ImageNet family; A33 says the order statistic is set by the reference draw; A31 says the level rises with K but the order does not respond to K. No clause of the title survives the package.

**41. MINOR — III-B presents a reversed prediction as verification.** "the registered sign was the opposite" (P9.10b/P9.11): a prediction that failed in sign is reported as "verified … before the profile settles". Report it as exploratory.

**42. MINOR — package §2 "every effect is reported next to the curve's own adjacent-layer jitter"** is not true of T–R (no CI), of the λ levels (0.42 → 0.078), or of the B10 widths.

**43. MINOR — the current manuscript's abstract and contributions (`main.tex` 48–75, 207–255) still carry "the kernel's gain aligns with its input's principal directions", "block alone does not produce it", "most families keep the ordering", "two thirds of the parameters" as headline claims that the package has withdrawn or demoted (A32 partial; A17/A26/A27 withdrawn; 3-D). §7.8 counts ≈ 60 pending corrections. A reviewer of the framework cannot tell which claims will survive into the text.**

**44. MINOR — the method half.** Fold-in = Chee 2022 / Yu & Wu (conceded); controller negative in 3 of 4 cells; the "sole surviving statement" *the ruler reads full ⇔ controller = uniform* is nearly a tautology (a shrink-only water-filling with nothing to shrink is uniform). Keeping this at a paragraph makes the paper neither a method paper nor a theory paper; the tool axis is empty for a TPAMI reader.

**45. MINOR — "the profile at initialisation is computable" means "simulate the random network on Gaussian images with the matched cross-power spectrum" (feasibility note A), single seed (seed 1) in the six-family test; the level has no closed form (3-A). Say "simulable from second-order statistics", not "computable".**

**46. MINOR — draft III-A(iii) claims Joudaki's predicted widening with depth "is not what the position-sampled covariance shows", but the plan states the pooled columns cannot test it (n/C) and the position-sampled object is not the one the theorem is about. This is not a measured hypothesis failure; drop it.**

**47. MINOR — transformers.** "Transformers belong to the order-keeping family" holds against the library-default draw and fails against the faithful initialisation (A33 P9.71: +0.52/+0.48, fc2 negative); attention proj at init is "almost one-dimensional (0.013)" (CLAUDE notes on A30), so rank correlations over near-degenerate init profiles recur here (finding 4).

**48. MINOR — pre-registration timestamps.** Registration-to-judgement intervals of 1–4 minutes (B8, B9, B10; git log 10-02 07:12–07:41, 10-06 07:20–07:22) will be read by any referee who checks the repository as pre-registration in name only. Either state plainly that these were checks on stored data with the hypothesis written down first, or stop using the word.

---

## (b) Answers to the five questions in §0

**1. Is the question big enough, and is the gap real?**
The question as posed — "what determines the representational width of every layer, how much is computable before training and predictable during it" — would be big if answered. The package answers the first half with a simulation (Gaussian surrogate through the random network) plus known mean-field results, and the second half not at all (κ undefined, finding 16). The gap as stated in §6 is overstated (finding 31): Harun 2024 and Masarczyk 2023 ask what sets the layer-wise rank profile and answer with the same factors; the genuinely open part is the position-sampled τ-width against random draws across families, which is a protocol contribution, not a question-sized gap. Verdict: the question is big; the delivered answer is not, and the gap that remains is a measurement gap.

**2. Which claims to strike / over-stated / known-as-ours?**
Strike: 3-D's "order lost at depth on ImageNet" as a property of networks (findings 1–3); P9.81/P-b9.7 as evidence (4); the A29b "turn between 20 and 36" (9); "three widths" with the accuracy cost (19); B10 as a chain link (24); the Ky Fan "principle" (18); "class count enters monotonically" in the chain (10); "not with image count" (39); the title candidate (40).
Over-stated: B9 as "the one quantitative statement that holds" (11, 12); B7 as a bound (23); ONI as counterfactual hit (26); the ledger as pre-registration (13, 14, 48); "computable before training" (45); the reproduction gates (33).
Known and mis-framed: M1 (30); B6 (17); the memory window (27); the direction of class-count and depth effects (10, 31); the alignment results (CRH/CNFA, conceded in 3-B but still "verified" in the draft).

**3. Is a chain of published theorems with measured hypothesis failure a theory contribution?**
No. It is a well-organised related-work section with measurements, and in two places (B6, the collapse "2–50 K") the "failure" is that the theorem was never about the measured object (findings 17, 25). A theory contribution would be a statement about κ, or a finite-width correlated-position law for the ReLU step (open, B4), or a proof of the order consequence (B1, abandoned). The package has none, and says so in §7.1.

**4. Most damaging objection and a cheaper framing** — see (c).

**5. Scores and probabilities** — see (d), (e).

---

## (c) The single most damaging objection, and the cheapest surviving framing

The central empirical phenomenon on which the chain, the title and the attribution experiments (A17, A28, A29b, A31 — roughly US$250 of compute) all rest — "the order of initialisation is lost in deep ImageNet networks" — is a property of the statistic, not of the networks. It vanishes when layers are ranked within a stage (public ResNet-50 within-stage T–R +0.48/+0.55/+0.57 for stages 1–3; A17 stage 1 +0.86/+0.87); it changes sign when the denominator is C instead of r_max (+0.57 vs −0.08 for the same six checkpoints); it changes sign with the reference draw (−0.25 vs +0.41 for the same trained weights); it is carried by the 1×1 layers where the r_max correction is largest; and in the ten-class ImageNet cell it is computed on a profile where 15–29 of 53 layers are tied at one channel. A referee who opens `checkpoint_analysis.txt` §6 finds all of this in the authors' own output. Once 3-D falls, 3-C's κ has no order to explain, the attribution-by-exclusion chain has no effect to attribute, and the title has no subject.

Cheapest framing that survives: a **measurement paper** with no order narrative and no theory section. (i) The protocol and the three artefacts that move published profiles (denominator, estimator, tensor), stated as corrections to the activation-spectrum literature with Garg 2024 credited for the denominator; (ii) the initial profile is reproducible from the architecture and the image cross-power spectrum by simulation (A7, with ≥ 3 seeds), and the layer-local Wishart law holds at finite width (535 layers); (iii) training raises the level in every family and reorders the *late stages* of deep ImageNet networks under r_max, reported within stage and within role, against both references, with CIs; (iv) the per-position cell structure of collapsed VGG-16/CIFAR layers (P·K − 1, blind at K = 13/20) as an exploratory observation; (v) the idle-width/fold-in results as a demoted application. That paper is honest, novel in its protocol, and fits a measurement-oriented venue (TNNLS, TMLR, or TPAMI as a short/measurement paper with the "theory" in a supplement). It does not need κ, B6–B10, or the 3-0 spine.

---

## (d) Scores (1 = rejection ground, 3 = neutral, 5 = reason to accept)

| Axis | Score | One sentence |
|---|---|---|
| Literature | 3 | Broad and recently thorough, but the gap sentence is overstated and several load-bearing attributions (K-FAC, Cui, "PCA-optimal", "Markov chain") are loose; M1 and the alignment results are known results in new clothes. |
| Size of the question | 2.5 | Big as asked; the delivered answer is "simulable at initialisation, not derivable after", and the one phenomenon the chain explains (order loss) is an artefact of the statistic. |
| Theory | 2 | One classical fact restated (B6), one linear-recursion lemma (B7.1), one identity (B10), one definition (B9.0); the stated "hypothesis failures" are in two cases comparisons with objects the theorems never addressed. |
| Data | 3.5 | A large, carefully gated, partly pre-registered measurement set with printed failures — undermined by n = 1 per family, two seeds per cell, no CI on the order statistic, a degenerate ten-class cell, and a stale primary analysis output. |
| Usable tool | 2 | The package implements a protocol; the fold-in is Chee 2022's operator; the controller is negative in three of four cells; "ruler reads full ⇔ uniform" is nearly a tautology. |

---

## (e) TPAMI outcome

- **Acceptance (after any number of rounds) ≈ 10–15 %.** The framework as submitted would draw at least one reject on findings 1–4 and 16–17 ("the headline phenomenon is an artefact of the statistic; the theory section contains no theorem"), and the author's own §7.1 concedes the theory point. A single-author measurement paper with ≈ 60 pending corrections and a title that the package contradicts does not survive TPAMI's first round.
- **Major revision ≈ 35 %.** Possible if the editor values the protocol and the ledger; the revision request would be: within-stage/within-role statistics with CIs and both references (findings 1–3, 5, 8), re-measurement of A31 (4), the stem-crossing cells (6), removal of the theory section to a supplement (16–26), one ledger tally with blind/non-blind labels (13–14), and a title that matches.
- **Reject ≈ 50–55 %.**
Reasoning: the paper's novelty is real but lies in the measurement protocol and the breadth of controlled trainings, which TPAMI referees weigh below a new result; the one claimed "result" (3-D) does not survive the authors' own within-stage and /C numbers; and the theory section, by the package's own description, is a chain of other people's theorems that mostly do not apply. The cheaper framing in (c) raises the major-revision probability to perhaps 50 % at a measurement-friendly venue, but not at TPAMI.
