# Framework review package (2026-10-02; English, self-contained; for an unanchored hostile review of the *claims*, before any rewriting)

Status of this file: assembled from `topic-statement`, `theory-chain-draft`, `claim-boundaries`, plan §10 and the ledger, with the judgements up to
2026-10-02 morning. Slots marked **[PENDING]** are filled when the runs finish (weight-decay sweep ≈ 16:00; A31 end cells ≈ 14:00; A29b second seed 10-03).
Nothing here is in `paper/` yet; the manuscript is rewritten only if this framework passes the review (stop-loss rule of 2026-09-19).

## 0. What the reviewer is asked

Venue: IEEE TPAMI regular paper (12 pages including references, supplement reviewed, single-blind). Single author, no students, rented GPUs only.
Review the *framework* — question, claims, evidence, theory — not the prose. Be hostile. Specifically:

1. Is the question big enough for TPAMI, and is the gap real given the nearest neighbours listed in §6?
2. Which claims would you strike, which are over-stated, which are already known and mis-framed as ours?
3. Is "a chain of published theorems, each with its measured degree of hypothesis failure, plus one small proposition" a theory contribution, or a literature review with measurements?
4. What is the single most damaging objection, and is there a cheaper framing that survives it?
5. Score five axes 1–5 (literature, size of the question, theory, data, usable tool) and give an acceptance probability with a major-revision probability.

## 1. The question

**What determines the representational width of every layer of a deep network — how many directions the spectrum of the layer's output covariance needs —
and how much of it can be computed before training and predicted during it?**

Width here is spectral: k*(τ), the number of principal directions of the channel covariance holding a fraction τ of the variance, divided by the attainable rank
r_max. It is not the channel count, not a manifold intrinsic dimension, not the numerical rank, not the participation ratio; the four are kept apart.

Title candidates (author prefers the third; the subtitle is unsettled because of §3-D):
*What Sets the Width of a Layer: The Spectrum of Deep Representations from Initialisation to Collapse* ·
*Layer Width Is Not a Hyperparameter …* (the method half has been demoted, see §5) ·
*The Layer-Wise Width Profile of Deep Networks: Initialisation Sets the Order, the Task Sets the Level*.

## 2. Object and protocol (the Method section)

- Channel covariance of a convolution's output (before normalisation and activation), sampled over 6,400 images × 16 positions; a layer is reported only if n/C ≥ 50.
- Denominator r_max = groups · min((C_in/groups)·k_h·k_w, C_out/groups); depthwise and dense convolutions never share a curve; conv output and block output recorded separately.
- Statistics are paired over all gated layers (Wilcoxon / sign test / rank correlation with depth); every effect is reported next to the curve's own adjacent-layer jitter.
- Profiles are compared by Spearman correlation over layers: T–R (trained vs a random draw of the same architecture), T–T (two trainings), R–R (two draws).
- 44 public checkpoints (22 trained + their random references), 11 CNN families and two transformer families; CIFAR and ImageNet controlled trainings.
- 162 pre-registered predictions in a ledger (supplementary table): 61 hold, 47 fail, 5 hold in reverse, 3 partly, 5 cancelled, 16 pending, the rest descriptive. Failures are reported.
- Two reproduction gates: Garg et al.'s CIFAR-10 VGG profile (13 layers, r = 0.981, MAD 0.052); Yu & Wu's DeiT-B fc1 spectra (12/12 blocks within 0.009).
- Package `layerspec` (PyPI 0.2.0, 180 tests) implements the protocol.

Measurement facts that need no theory (all survive paired tests on every gated layer):
(M1) normalising by C instead of r_max manufactures a "bottleneck deficit" in 14/14 models with 1×1 expansions; by r_max the bounded/unbounded shift is 3 % of that
(median residual −0.018 [−0.028, −0.004] of r_max). The bound is known (Han 2021; Kim 2018) and is Garg 2024's denominator; ours is the opposite conclusion for the
C-normalised activation-spectrum literature and the group factor. (M2) pooling lowers the level in 687/703 dense and 79/80 depthwise layer pairs; the drop exceeds the
curve's own jitter in 18/19 models (median ratio 2.2). (M3) the block output exceeds the conv3 bound in 161/161 bottlenecks (ratio ≈ 4): r_max bounds convolution outputs, not the stream.
(M4) trained ≠ random in ResNet-50 (six recipes, all p ≤ 3·10⁻⁸); in ConvNeXt only for dense layers (p = 0.013; depthwise 0.21).

## 3. The chain (the Theory section): five links, each = published theorem + hypothesis + measured degree of failure + what we verify

### 3-A. Initialisation: the profile is a function of architecture, initialisation scale and the image second-order statistics
- Known: position-covariance recursion (Xiao 2018; Poole 2016); exact finite-width law of the linear step (Silverstein 1995); BN fixed point / isometry (Yang 2019; Joudaki–Daneshmand–Bach 2023);
  PR ≥ Ω(√C) under BN (Daneshmand 2020).
- Hypotheses that fail for our object: Gaussian pre-activations (the ReLU step at τ = 0.999 needs the full per-channel marginals: 13/13 layers within 0.03 with them, off by 0.03–0.12 with four cumulants;
  no closed recursion found); **sample independence** (our samples are positions of one image — the case the theorems exclude); this is the real hole.
- Verified: the layer-local law reproduces 535 layers of 11 families at initialisation with median error ≤ 2 channels (94–100 % of each family within 0.05 r_max); the profile's law depends on the images
  only through the first-layer patch Gram (proved, elementary); a surrogate built from the image spectrum alone reproduces order (ρ 0.95–1.00) and level (median +0.046) in six families (seed 1);
  swapping the ImageNet spectrum for CIFAR-10's moves the level by 0.06–0.24 (the spectrum is the causal channel).
- The initialisation *scale* is an input: zero-γ versus γ = 1 in the last BN of a block changes the random level of ResNet-18 from 0.29–0.34 to 0.44–0.48 and R–R from 0.98 to 0.71;
  what decides T–R is the reference draw, not fidelity to the weights' true starting point (A33: P9.68/69 hold).
- Failed/limited: no closed form for the *level* (three closed forms over/undershoot by 0.05–0.2); "identity shortcuts keep the input spectrum" fails (ResNet-18 keeps it, ResNet-50 forgets it like
  plain networks; P9.86, three seeds); an a-priori starting width gave no accuracy gain in two architectures.

### 3-B. Early training: alignment precedes the width change
- Known: silent alignment (Atanasov 2022), spike + bulk (Ba 2022), convolutional neural feature ansatz (Beaglehole 2023; Radhakrishnan 2024), FACT (Boix-Adserà 2025), first-layer PCA alignment (Maennel 2020),
  canonical representation hypothesis — six alignments among representation, weight and gradient second moments (Ziyin et al., ICLR 2025).
- Verified: kernel gain aligns with the input principal directions (ρ_align ≈ +0.9 in 19 ImageNet weights; random 0.0) and does so by epoch 7–15 of 100 in 9/9 runs, before the profile settles;
  the registered sign was the opposite (reported as a reversed prediction). The gain follows the output-side gradient energy first (Spearman +0.98/+0.95 — Ziyin et al.'s forward gradient–weight
  alignment, here on conv patches) and the input variance second (partial +0.55/+0.45); the element-wise correlation used by the CNFA literature has a structural floor of 0.8 (channel permutation leaves it unchanged).
- Counterfactual that hits: forcing exact kernel isometry (ONI, T = 20) moves the profile by the predicted amount (|out − pred| 0.021–0.031; realised 0.91–1.04; two architectures × three seeds).
  A soft orthogonality penalty realises only half (kernel erank 0.84).
- **Negative (2026-10-02, registered before reading spectra):** the polynomial-alignment form of Ziyin et al. (Σ_out ∝ Γ² under their phase 8) does *not* extend to position-resolved conv–BN layers:
  Σ_out and the output-gradient Gram do not share an eigenbasis (co-diagonal fraction 0.17–0.23 against 0.09–0.15 for a random basis), exponents are not the phase-8 values (gain ∝ γ^{2–5}); P-b8.1–8.4 all fail.
  The alignments hold in rank only.

### 3-C. Late training: collapse theory bounds a layer's rank at ≈ K — and what we measure is 2–50 K
- Known: at a stationary point with weight decay, low within-class variation ⇒ weights near rank K (Zangrando 2026, Thm 3.5); deep UFM optima of rank Θ(√K)…K (Súkeník 2024), ≤ 2 for large depth (Garrod–Keating 2025);
  depth alone biases to low rank (Jacot 2023); collapse in the terminal phase (Papyan 2020). All predict a late rank at or below K.
- Measured: late τ-width on CIFAR-10 is 20–50 channels for K = 10; numerical rank 68–138; ten-class ImageNet ResNet-18 reads 0.95 of 512 channels.
- **Proposition B6 (ours, small, exact):** for conv followed by training-mode BN, ⟨∇L₀, W⟩ = 0, so λW = −∇L₀ has no solution with W ≠ 0: Theorem 3.5 has no stationary point to bite on.
  (Scale invariance is classical; Chen–Bu–Yan 2024 observed the zero cosine at initialisation.) Verified on 21 checkpoints × 13 layers: |cos| ≤ 0.043, gradient 1.3–13× λW, rank-K tail energy of late weights 0.41–0.62,
  while the pooled post-ReLU features *do* reach NC1 at the last convolution (within-class fraction 0.07): collapse is a property of the pooled representation, not of the weights or the position-sampled width.
  Two rulers on the same network (numerical rank vs k*(0.999)) move together and neither reaches K (A34).
- **B7 (memory window; the window itself is known — Wang & Aitchison 2025):** exact for SGD with momentum and any schedule, W_T = a_T W_0 − Σ c_s g_s with c_s ≥ 0 (proved; verified to 10⁻¹⁰);
  hence col(W_T) ⊆ span of position-resolved output gradients over the window, and Σ_out = Δ̃ M Δ̃ᵀ against Γ_win = Δ̃Δ̃ᵀ (two-sided τ-width bound with constant cond(M)).
  Verified: k*(Σ_out) follows k*(Γ) over 21 checkpoints (Spearman +0.86); Γ ≥ 3K wide everywhere; Γ narrows with optimisation steps (310 → 108); a forward-only formula (ReLU masks × EL2N) reproduces the *order*
  of Γ's head (ρ = 1.000 in 21/21) but over-counts by 1.6–2.1×; the window Gram orders the output widths within a network (+1.00; +0.83–0.88 after dividing by C).
  **Not verified:** the bound is tight only where the layer is already at its ruler width; on the full-width network the late window Gram is 12× wider than Σ_out. B7 is an inclusion and an ordering, not a bound.
  κ follows the fraction of hard examples (Spearman +0.85) but is not separated from K (partial +0.58). **There is no derivation of κ.**
  Weight-decay sweep **[PENDING: P-b7.4a–d]**; window-versus-end exponent **[PENDING: P-b8.5/8.6]**.
- **B8 (negative; scopes Ziyin et al. 2025):** the alignment hypothesis would give κ = k*_τ(Γ^α) with α ≈ 2. Registered before the check, on existing data: Σ_out and the end-point gradient Gram do not share an eigenbasis
  (co-diagonal fraction 0.17–0.23 against 0.09–0.15 for a random basis), the exponent is 0.12–0.76, not 2 (P-b8.1–8.4 all fail). The hypothesis does not extend to position-resolved conv–BN layers.
- **B9 (collapse is per position; the one quantitative statement that holds):** Lemma (elementary: total-variance decomposition + Ky Fan): if the variance outside the span of the (class × position) cell means is ≤ 1 − τ,
  then k*_τ ≤ P·K − 1. Predictions committed before the spectra were opened: in trained ten-class VGG-16 on CIFAR-10 the spectral cliff sits at 39/40 = 4 positions × 10 classes in 18/18 layers of the last block
  (ratio 10–77) and k*(0.999) ≤ 40; blind at K = 13 and 20 the cliff is at 51/52 and 79 in 9/9 layers that have one. Cell means are linearly independent (rank 38–39 of 39, 18/18); class-only cells leave 79–95% of the variance
  (position is necessary, 15/18); the within-cell variance is 2–25% but lies inside the span (outside fraction 0.0006–0.0012, straddling 1 − τ, hence widths 39 and 40).
  **Scope:** measured only in VGG-16 on CIFAR. The ten-class ImageNet ResNet-18 exceeds 490 = 49·10 in 11/12 last-stage sites (P-b9.8 fails): no per-position collapse there. K = 50/100 show no cliff.
  A prediction from the cap (ten classes at 224 px do not bind ResNet-50's last stage, so the order is not kept) held against the registered opposite (P9.81); its second clause failed in one of three sites per seed.
- Class count enters monotonically: the ruler keeps 0.88/0.92/0.98/0.99 of the parameters at K = 13/20/50/100 (3/3 seeds); the contraction deepens with optimisation steps, not with image count.
  Direction known (Masarczyk 2023; Harun 2024); ours is the per-layer number at fixed τ and the dose–response.

### 3-D. Net effect on the layer order — the weakest link
- Statement: initial profiles fall with depth; training raises every level and caps the late layers at κ; with few classes the cap is low and the order survives, with many classes at depth the late layers rise and the order is lost.
- Evidence for: CIFAR-10 families T–R +0.4 to +0.9; ImageNet ResNet-50 and a 52-layer basic network under one recipe T–R −0.25/−0.26 and −0.21/−0.02; recipe excluded (T–T +0.86 against six public recipes);
  block type excluded (H9.4a fails); within ImageNet neither 100 classes, nor 64 px, nor 130 images/class restores the order, and 10× more steps pushes further (+0.02 → −0.32) (A28).
  Depth under one recipe: ResNet-18 +0.75 (keeps), ResNet-34 +0.01, 52-layer −0.11 **[PENDING: second seed, P9.50]**. Ten classes at 224 px do not restore the order: same-seed T–R +0.03 / +0.10 (P9.81 fails, two seeds); a hundred classes −0.29 on one seed **[PENDING: second seed, P9.82]**.
- Corrections already forced by an independent review: every T–R names its reference initialisation (the same ResNet-50 weights read −0.25 against γ = 1 and +0.41 against γ = 0);
  all-layer T–R in families that alternate 1×1 and 3×3 is a two-population artefact (DenseNet-121 +0.71 overall, −0.26 within 3×3); within role, 3×3 order is kept in VGG-16/ResNet-18/34 and lost in every deep ImageNet family.
  "Block type decides" and "plain versus residual" are both withdrawn (A17, A26, A27). Attribution stops at "the data at this depth, and optimisation length".
- Known directions (Harun 2024; Masarczyk 2023): class count, resolution, width, image count all reported with probes/numerical rank. Ours: the T–R statistic, exclusions, dose–response.

### 3-E. Residual streams
- Known: Weyl; rank preservation at initialisation (Dong 2021; Joudaki–Hofmann 2024); low-rank branches after training (Li & Papyan 2023).
- Verified on 56 blocks of four ImageNet ResNets: every eigenvalue of the stream is larger after the addition (retention ≥ 1.11), yet the τ = 0.95 width before the ReLU is narrower in 36/42 blocks
  (absolute versus relative statement). Our candidate lower bound on the τ-width (B2) holds but reaches only 0.24 of the measured width — closed, not in the paper.

### Transformers (same ruler)
ViT-S/ViT-B: T–R +0.69/+0.61 over all layers and 0.83–0.98 within role against the library-default draw; T–T +0.90/+0.89; level 0.16 → 0.43–0.58. Against the faithful initialisation (JAX / LayerScale 1e-4)
T–R is +0.52/+0.48 and fc2 is negative (A33). Yu & Wu's fc1 reading reproduced. Prior: Yu & Wu 2023, Garg 2024 give per-block spectra; ours adds random references, T–T, roles, the gate.

## 4. What is ours, stated plainly

1. The object (layer-wise τ-width profile of position-sampled channel covariance, normalised by r_max, gated) and the protocol facts M1–M4.
2. The assembly: no line of work states what sets the layer-wise profile; each link cites its theorem and reports *how far its hypotheses fail on convolutional networks* (535-layer law test; copula diagnostic;
   sample dependence; BN removes the stationary point; pooled-versus-position collapse; rank-only alignments).
3. One exact proposition (B6), one exact lemma made sharper than the known EMA view (B7.1, with the non-negativity condition), one elementary identity (layer-local law at finite width), all with numerical tests.
4. Controlled experiments that exclude attributions (recipe, block type, plain/residual, class count within ImageNet, resolution, image count) and one counterfactual that hits (ONI).
5. A ledger of 162 registered predictions with the failures printed.

## 5. Demoted (one paragraph each in the paper, or supplement)

- Width read from the covariance used as a method: zero-retraining fold-in keeps VGG-16/CIFAR-10 accuracy at 67 % of the parameters (93.82 vs 93.83; three seeds −0.04); on thousand-class networks at τ = 0.999 there is nothing to remove (1–3 %).
  Same operator as Chee 2022 / Yu & Wu 2023 — not an operator contribution.
- One-shot controller ("shrink-only water-filling"): beats uniform only on VGG-16/CIFAR-10 (+0.24–0.40); equals uniform on ResNet-20 (+0.13), ×4-wide ResNet-20 (−0.05; no idle width appears on widening, P-l2 fails) and a CIFAR ViT.
  The grow operator contributes nothing measurable (P-l1). Sole surviving statement: *the ruler reads full ⇔ controller = uniform*, four cells.
- ImageNet ResNet-18 trained at ruler widths is a null test (the ruler reads 99.2 % of the parameters).

## 6. Nearest neighbours and why the gap is claimed to be real

Zhu–Zhao–Lu 2026 (NSC: architecture spec → a scalar of the *weight* spectrum); Liu–Paquette–Sous 2026 (Spectral Lens: activation and gradient spectra side by side in LLM training);
Garg 2024 (utilized rank per layer ÷ min(m, d)); Yu & Wu 2023 (DeiT per-block spectra, low-rank fold-in); Masarczyk 2023 / Harun 2024 (tunnel effect, factors);
Ziyin et al. 2025 (representation–weight–gradient alignment; fully connected layers); Chen–Bu–Yan 2024 (per-layer "wide enough" criterion from weight-norm variance);
Bayat 2026 / Mudarisov 2026 (layer-wise MLP widths in LLMs). A citation-graph scan (Ansuini 457 citing papers, Feng, Recanatesi, Daneshmand, …) found none that asks what sets the layer-wise profile or computes it before training.
Self-admitted in the literature: "existing studies are exclusively empirical" (Yusupov 2025).

## 7. Weaknesses we already know (do not spare them)

1. **No new theorem of substance.** B6 is elementary; B7.1 sharpens a known fact; B7.2′ is an ordering; κ has no derivation; B8 (the one attempt at a quantitative law for κ) failed as registered.
2. **Sample independence** fails for every random-matrix theorem in 3-A; the activation step has no law for correlated positions.
3. **3-D is an attribution by exclusion** that ends in "the data at this depth"; the directions are in Harun/Masarczyk; T–R depends on the reference initialisation and on the role mix.
4. Most families have n = 1 trained checkpoint (only ResNet-50 has six recipes); layer-level detail outside ResNet-50 has no error bars.
5. The theory explains *order* (rank correlations) in most places, *numbers* only at initialisation's linear step.
6. Practical value is thin: idle width exists only in few-class plain networks; the controller is negative in three cells of four.
7. CNN-centric with two transformer families bolted on; 2026 readers may ask for LLMs.
8. Several headline numbers of the current manuscript are superseded and the text has ≈ 60 pending corrections (lists exist).

## 8. Pending slots (filled before the review runs)

- **[PENDING] P-b7.4a–d** weight-decay sweep: does λ set which gradients the width is read from, does the late level fall with λ.
- **[PENDING] P-b8.5/8.6** window-Gram exponent ≈ 1 versus final-Gram 0.3–0.5 on four unseen runs.
- **[PENDING] A31 K100 seed 1** (P9.82): the final files are on the network volume; K10 is judged (P9.81 fails).
- **[PENDING] A29b seed 1** (P9.50): depth axis 20/36/52 under one recipe.

## 9. Self-score before the review (author's five axes; 1 = rejection ground, 3 = neutral, 5 = reason to accept)

| Axis | 10-01 | 10-02 morning | Why |
|---|---|---|---|
| Literature | 3.5 | 3.5 | CRH/PAH paper had not been read (found today, read in full, bounded); boundary sentences exist for 14 claims, none in the text yet |
| Question | 3.5 | 3.5 | the "computable after training" half did not become a bound; the question stands, the answer is half |
| Theory | 3.5 | **3.5 (ceiling 4 on this route)** | P-b7.1′ holds only vacuously; B8 failed; what remains is B6 + inclusion/ordering + measured hypothesis failure |
| Data | 4 | 4 | A29b/A31 pending |
| Tool | 3.5 | 3.5 | unchanged |
