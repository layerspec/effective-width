# Theory-chain section — first paper-style draft (2026-09-30 afternoon; English; NOT yet in `paper/`; gated by the framework review)

Purpose: turn `theory-chain-2026-09-30.md` §A into the five subsections the paper will carry (axis 3, step C.2). Every link states
(a) the theorem as published, with its hypotheses; (b) how far those hypotheses hold for the object we measure (conv output,
channel covariance over positions, n/C ≥ 50, k*(τ)/r_max); (c) what we measured that verifies or bounds the link.
Numbers marked [check] must be re-read from the results files before they enter `paper/`. Exploratory items are labelled.
Nothing here is a new theorem except Proposition B6 (small) and the residual τ-width inequality B2 (elementary); both are
stated as propositions with proofs in the supplement.

---

## III. What sets the width of a layer: five links

We measure one object throughout: the channel covariance Σ_l of the output of convolution l, sampled over images and positions,
summarised by k*(τ), the number of principal directions holding a fraction τ of the variance, divided by the reachable rank r_max.
The literature on deep representations contains, spread over five lines of work, everything needed to say what determines this
quantity at initialisation, how training moves it, and what bounds it at the end. No one of those lines states the layer-wise result;
we assemble it here, one link per subsection, and state for each link the hypothesis under which it is a theorem and the measured
degree to which that hypothesis holds for convolutional networks.

### III-A. Initialisation: the spectrum is a function of architecture, initialisation scale and the image second-order statistics

**Known.** For i.i.d. Gaussian weights the pre-activation of layer l+1 at a set of positions is, conditional on layer l, Gaussian with a
position covariance obtained by convolving the post-activation covariance of layer l with the kernel's spatial support
(Xiao et al. 2018, eq. (2.7), Σ^{l+1} = A ⋆ C(Σ^l), Lemma A.1 for c → ∞; the fully connected case is Poole et al. 2016 and Schoenholz et al. 2017). The map from Σ_l to Σ_{l+1}
therefore factorises into a *linear* step, whose spectrum is given exactly at finite width by the Silverstein (1995) law for
W̃ Σ W̃ᵀ with W̃ Gaussian (our Proposition S6), and a *non-linear* step (the activation), for which the mean-field literature gives the
infinite-width covariance map (the C-map of Poole et al.) and the linear-width literature gives a random-matrix law under
independence assumptions (Fan and Wang 2020; Louart, Liao and Couillet 2018). With batch normalisation the recursion has a
stable fixed point at which the batch Gram is isometric (Yang et al. 2019, BSB1 fixed point; Joudaki, Daneshmand and Bach 2023,
Theorem 4: the distance to isometry decays as exp(−ℓ log β_σ) with β_ReLU = 2 − 2/π), and its finite-width deviation is bounded by
exp(−α·depth) + n/√d (Joudaki, Daneshmand and Bach, ICML 2023, Theorem 1, under a geometric-ergodicity assumption the authors
leave open). The scale of the initialisation enters the same recursion (γ of the last BN in a block, LayerScale in transformers;
Kunin et al. 2021 for the dynamics it induces).

**Hypotheses and how far they hold.** (i) *Infinite channels.* We measure at C = 64–2048; the layer-local law (Prop. S6) is exact at
finite C for the linear step, and on 535 dense convolutions of 11 families its median deviation from the measured k* is ≤ 2 channels (≤ 0.016 r_max) at both thresholds, 94–100 % of the layers of each family within 0.05 r_max, rank correlation ≥ 0.974 [supp. Tables S18–S19]. (ii) *Gaussian pre-activations across positions.* Violated: the marginals have
excess kurtosis, and the copula diagnostic shows that at τ = 0.999 the ReLU step is determined by the Gaussian copula together with
the *full* per-channel marginals (13/13 ReLUs of VGG-16 within 0.03), but not by marginals matched to four cumulants (misses by 0.03–0.08), and at τ = 0.95 the deep layers need the pairwise dependence as well; the closed forms (arc-cosine recursion, Gaussian equivalence, linear term) over- or under-shoot by 0.05–0.2 [supp. law paragraph; feasibility note C.1–C.3]. (iii) *Independent samples.* The theorems of Fan–Wang, Louart et al. and both Joudaki papers
take the n samples independent; ours are positions of the same image, and neighbouring patches are the "duplicated samples" that
Joudaki et al. 2023 explicitly exclude. This is the hypothesis that fails most, and it is why the predicted *widening* with depth of a
normalised network (Gram → identity) is not what the position-sampled covariance shows [exploratory, seed 1: ρ(depth, k*₀.₉₅/r_max)
between −0.3 and +0.8 across ten families, none monotone to full width].

**What we verify.** (1) The profile at initialisation is computable from the architecture and the image covariance alone: on six
families the surrogate built from the image spectrum alone reproduces the measured profile in order (ρ 0.95–1.00, at or above the random–random ceiling) and level (median difference +0.046) [A7, feasibility note; seed 1].
(2) The image statistics enter only through the first-layer patch Gram (Prop. S7), and they matter: replacing the ImageNet spectrum
by the CIFAR-10 spectrum at the same resolution lowers the level by 0.06–0.24 and, in VGG-16 and ResNet-50, the order to ρ 0.67/0.74 against ceilings of 0.79/0.94 [A7, P-a1 6/6]. (3) *Exploratory (seed 1):* the effect of the
image spectrum decays with depth in plain and dense families (relative difference in the last third 0.00–0.24) but not in ResNet-18/50
(0.4–0.65 in every third) — the "input forgetting" of Joudaki et al. and the slowing of it by residual connections predicted by
Joudaki and Hofmann (2024, Prop. 2: κ_res = (1−r²)κ + r²ρ). (4) The initialisation scale is an input, not a nuisance: zero-initialising
the last BN of each block (timm) versus γ = 1 (torchvision) changes the random reference level from 0.29–0.34 to 0.44–0.48 in
ResNet-18 and the random–random agreement from 0.98 to 0.71, while the trained-vs-random rank correlation is unchanged
[check: §10.16; A33 pending for ViT/DeiT/ResNet-50].

*Conclusion of the link.* The order of the initial profile is an architectural quantity, its level is set jointly by the architecture,
the initialisation scale and the image spectrum; the theory that predicts it is exact for the linear step and only approximate for the
activation step, with the failing hypothesis being sample independence.

### III-B. Early training: alignment precedes the width change

**Known.** In deep linear networks the input–output singular modes are learned in order of singular value (Saxe et al. 2014); with
small initialisation the kernel's eigenstructure aligns with the task before the loss falls ("silent alignment", Atanasov et al. 2022);
one large gradient step adds a rank-one spike aligned with the target to an otherwise unchanged bulk (Ba et al. 2022), and at small
learning rate the bulk of the conjugate kernel is invariant (Wang et al. 2023, Lemma 5.1/Theorem 5.2). For every layer of a trained
CNN the Gram of the kernel, WᵀW, is proportional to the gradient outer product of its input patches (the convolutional neural feature
ansatz, Beaglehole et al. 2023; Radhakrishnan et al. 2024), and at convergence with weight decay this proportionality is a first-order
condition (Boix-Adserà et al. 2025). The first layer aligns with the data principal components (Maennel et al. 2020). In transformers the
rank of W_t − W_0 grows gradually (Boix-Adserà et al. 2023).

**Hypotheses and how far they hold.** Small initialisation and whitened data (Atanasov), two layers or linear width (Ba, Wang),
convergence with weight decay (Boix-Adserà 2025) — none holds for a standard-initialised, batch-normalised CNN under SGD; and in a
conv–BN layer the first-order condition of FACT cannot hold at all (Prop. B6 below).

**What we verify.** The gain of each kernel aligns with the principal directions of its input covariance (ρ_align ≈ +0.9) and does so by
epoch 7–15 in 9/9 runs, before the profile settles (epoch 15–100) [§10.7: P9.11 registered sign fails, reverse holds 7/9; 19 ImageNet weights median +0.91, six ResNet-50 recipes +0.86 to +0.95; inverted-residual families +0.11 to +0.61]; the numerical rank of the late layers rises
from 0.11 to 0.72 of C within the first 25 steps [check: A34 part b]. The alignment is only partly a shadow of the neural feature ansatz:
the gain follows the output-side gradient energy first (Spearman +0.98/+0.95) and, holding that fixed, the input variance second
(+0.55/+0.45) [A32, §10.21]. The counterfactual holds: forcing exact isometry of the kernel (ONI, T = 20) moves the profile by the
amount the decomposition predicts (|out − pred| 0.021–0.031, realised fraction 0.91–1.04, two architectures, three seeds) [D5].

### III-C. Late training: collapse bounds the rank of a layer — and why the bound does not bite in batch-normalised layers

**Known.** With weight decay, every stationary point of the regularised loss satisfies λW_l = −∇_{W_l}L₀ (eq. (4) of Zangrando et al.),
and if the class means of the features entering layer l have small total within-class variation TCV_K, the distance of W_l to the set of
rank-≤K matrices is at most M_{l,j}·TCV_{K,j}/λ (Zangrando et al., TMLR 2026, Theorem 3.5). In the deep unconstrained-features model
the global optimum is not deep neural collapse but a strongly-regular-graph solution whose intermediate features and weights have rank Θ(√K) before the ReLU and K after it, for K ≥ 6 with L ≥ 4 layers or K ≥ 10 with L = 3, with a loss ratio O(K^{(3−L)/(2(L+1))}) (Súkeník, Lampert and Mondelli 2024, Theorem 5 and Definition 4), of rank at most q_K ≤ 2 for large depth (Garrod and Keating 2025, Theorems 2–3),
and deep neural collapse persists as a positive-semidefinite critical point whose share of the landscape grows with width
(Garrod and Keating, Theorem 5); the same optimality carries to regularised ResNets and transformers (Súkeník et al. 2025). Depth alone
biases toward low rank: at any global minimum of the L2-regularised loss some hidden representation has s₂/s₁ = O(L^{−1/4}), and the limiting representation cost per layer lies between the Jacobian rank and the bottleneck rank of the function (Jacot 2023, Proposition 4 and Theorem 1; fully connected, homogeneous activations), and the collapse happens in the terminal phase of training (Papyan, Han and Donoho 2020).
Every one of these results predicts a late-layer rank *at or below K*.

**What we measure is above K by a factor 2–50.** The late-layer τ-width on CIFAR-10 is 20–50 channels for K = 10 [check: A18/A24], the
numerical rank (eigenvalues above 10⁻³λ₁) 68–138 channels [A34], and a ten-class ImageNet ResNet-18 reads 0.95 of its 512 channels
[A26]. Two explanations are available and we test the first.

**Proposition B6 (no stationary point to bound).** Let layer l be a convolution followed by batch normalisation in training mode, so that
L₀(cW_l) = L₀(W_l) for all c > 0. Then ⟨∇_{W_l}L₀, W_l⟩ = 0, and the only solution of λW_l = −∇_{W_l}L₀ with λ > 0 is W_l = 0.
*Proof.* Differentiate L₀(cW_l) at c = 1; take the inner product of the stationarity equation with W_l. ∎
*Corollary B6′.* A stationary *direction* of the weight, (∇L₀ + λW) ∥ W, requires ∇L₀ = 0 exactly.
The scale invariance is classical (van Laarhoven 2017; Li and Arora 2020, Lemma 1.3; Zhang, Wang and Xu 2019; Kunin et al. 2021) and the
resulting zero cosine was observed at initialisation by Chen, Bu and Yan (2024: |cos| ≤ 3·10⁻⁴ over all channels of VGG16_bn) and
attributed to high dimension; what B6 adds is that it is exact, holds along the whole trajectory, and removes the hypothesis of
Theorem 3.5 for every conv–BN layer. Training then reaches the balance between weight decay and gradient rotation (the effective-step-size
equilibrium of Li and Arora), in which the norm, not the rank, of W is set by λ.

**What we verify [A35, pending].** On 21 CIFAR checkpoints (VGG-16_BN, K = 10/13/20/50/100, three seeds each): the training-mode cosine
per conv–BN layer, the stationarity residual ‖λW + ∇L₀‖/‖λW‖ (a smoke run gives 16–28: the trained weights are far from (4)), the rank-K
tail energy of the late weights (smoke: 0.41–0.61 — far from rank K), and the within-class variance fraction along depth
(smoke: 0.86 → 0.07 on pooled features — NC1 *is* reached in the pooled representation). The second explanation — that collapse is a
property of the pooled/flattened representation whereas the position-sampled covariance is a different object — is consistent with the
last number and with A34 (both rulers contract with optimisation steps on the position-sampled covariance, neither reaches K).

**The class count enters where the theory says it should.** The width kept by the ruler rises monotonically with K
(0.88/0.92/0.98/0.99 for K = 13/20/50/100, 3/3 seeds) [A24], contracts with optimisation steps at fixed data (d = 0.38 vs 0.52 at equal
steps) [A25], and a ten-class ImageNet task at 224 px shows no idle width in a residual network [A26] — so the few-class collapse of the
late layers is a property of plain networks on few-class tasks trained to the terminal phase, and its depth is set by κ, which we can
measure but not yet derive.

### III-D. Net effect on the layer order: reversal is the assembled consequence of III-A and III-C, not a separate phenomenon

At initialisation the profile falls with depth (III-A): ρ(depth, k*) at epoch 0 is −0.46 to −0.91 in the five trajectory families (VGG-16, basic52, bottleneck on CIFAR-10; basic6666 and ResNet-50 on ImageNet, two seeds each) [B1′ table]. Training raises every layer's level
(III-B) and then caps the late layers at κ(K, resolution, …) (III-C). If κ lies below the final level of the early layers, the order is
preserved (CIFAR-10, K = 10: late third 0.03 → 0.04, early third 0.42 → 0.31; T–R +0.40 to +0.89); if the late layers are uncapped they
rise from the narrowest to the widest and the order reverses (ImageNet, K = 1000: late third 0.07 → 0.56, early third 0.48 → 0.44;
T–R −0.21 to −0.26 in ResNet-50 and a 52-layer basic-block network of the same recipe) [A17, round 3]. The recipe is excluded
(this recipe against six public ResNet-50 recipes: T–T +0.86) and the block type is excluded (basic and bottleneck reverse alike on
ImageNet and preserve alike on CIFAR-10); what remains is the dataset at this depth [A17; A28 pending decides between class count,
image count and resolution]. No theorem in the literature states the layer order; none is needed beyond III-A and III-C, and the one
open quantity is κ.

### III-E. Residual streams: the addition cannot lower the spectrum, only the τ-width

**Known.** For Σ_y = Σ_x + Σ_f + cross terms with ‖cross‖ controlled by ρ_max, λ_i(Σ_y) ≥ λ_{i+r}(Σ_x) (Weyl) and
Σ_y ⪰ (1−ρ)(Σ_x + Σ_f) — classical; the mechanism that residual connections keep rank at initialisation is Dong et al. (2021) for
attention and Joudaki–Hofmann (2024, Prop. 2) for the kernel map; after training the branches are low-rank (Li and Papyan 2023).

**Proposition B2 (τ-width at the addition) — status 09-30 afternoon: proved only for an uncorrelated branch; with the (1±ρ) correction it is vacuous on all 56 measured blocks (ρ_max ≈ 0.95), so it stays out of the paper until a cross-term version is checked.** What can be said without it: the absolute spectrum cannot fall (Weyl), the relative width can, and in ResNets it does (below).

**What we verify [§10.17].** In 56 blocks of four ImageNet ResNets every eigenvalue of the stream at its own width is *larger* after the
addition (retention ≥ 1.11, median 1.36–5.77), both lower bounds hold in every block, and yet the τ = 0.95 width before the ReLU is
narrower than the stream's in 36 of 42 non-downsampling blocks — the gap between an absolute and a relative statement. The block output after
the ReLU exceeds the conv3 upper bound r_max in 161/161 bottlenecks [round 2], so the r_max bound is a statement about convolution
outputs, not about the stream.

---

## What this section does and does not claim (for the framework review)

- Claims: each link is a published theorem or a measured fact; the assembly (III-D) is ours; Proposition B6 is ours and small (B2 withdrawn pending a cross-term version);
  the measured *degree of hypothesis failure* per link is ours and is the part reviewers cannot find elsewhere.
- Does not claim: a derivation of κ; a finite-width, correlated-position law for the ReLU step (open; B4); that the order reversal is a
  theorem (it is a consequence under the stated conditions, verified on two ImageNet architectures and one CIFAR family).
- Weakest points: III-A(iii) sample independence (the real hole); III-C rests on A35 (pending); III-D's "dataset at this depth" awaits A28.
