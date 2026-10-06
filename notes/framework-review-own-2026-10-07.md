# Framework review, my own (2026-10-07; written before reading the independent agent's review; against `framework-review-package-2026-10-02.md` as of commit 4312fcb and `theory-chain-draft-2026-09-30.md` as of afaf8f5)

Stance: a TPAMI referee who has read the five lines of work and wants a reason to reject. Each finding: severity, claim, objection, where, fix.

## A. Findings

**A1 (FATAL as a theory paper, MAJOR as a measurement paper). There is no theorem of substance, and the new spine makes this visible.** The 3-0 spine says every width bound in the paper is Ky Fan's maximum principle on a particular covariance. That is true, and a referee will read it as: the propositions are instances of one textbook inequality (truncation error = discarded tail; cell-mean cap = rank of a span; bias direction = convexity of a sup of linear functionals). B6 is the classical scale invariance of BN stated as a non-existence of stationary points; B7.1 is an impulse response; B10 is an identity. The paper cannot be submitted as a theory contribution. *Fix:* say so in the first paragraph of the theory section — "the theory of this paper is the correct application of known results; the contribution is the measured degree to which their hypotheses fail on convolutional networks, and one chain that no line of work states" — and move the proofs to the supplement with that label.

**A2 (FATAL for 3-D as stated). The mechanism statement of 3-D is not testable in its own terms.** "If κ lies below the final level of the early layers, the order is preserved; if the late layers are uncapped they rise and the order is lost." κ is never computed before the fact (the package says so three times), so the antecedent is read off the same profile whose order it is supposed to explain. As written, 3-D is a description of two regimes with a post-hoc label, and A31 (K = 10 at 224 px reorders; response not monotone in K) shows the one knob that was supposed to move κ does not move the order. *Fix:* demote 3-D to "what we observe and what we excluded" (recipe, block type, plain/residual, class count, resolution, image count), state the one positive regularity (depth under one recipe: the turn between 20 and 36 basic convolutions, P9.50) and stop. Do not write "the cap sets the order".

**A3 (MAJOR; CONFIRMED and worse: four sources. `scripts/prereg_counts.py` on Table S17 gives 164 rows: 77 hold, 56 fail, 5 reversed, 2 by letter, 6 partly, 7 reported only, 4 cancelled, 4 never run, 3 not found; the package now quotes this; the ledger summary block is replaced by a pointer to the script.)** Three different counts of the pre-registration ledger coexist. The ledger's own summary block (`preregistration-ledger-2026-09-25.md` §統計) says 77 rows, 22 hold, 18 fail, 22 pending; the table under it has 162 rows with (by column) 65 hold, 53 fail, 5 reverse, 5 cancelled, 4 pending; the package §2 says 162, 61 hold, 47 fail, 16 pending. Table S17 will be read against the text; a referee who counts will find three numbers. *Fix:* regenerate the summary block from the table by script, make `table_prereg.tex` and the package quote that script's output, delete the hand-written block.

**A4 (MAJOR). "The width a PCA-optimal encoder would use" is not what ONI measures.** ONI makes the *kernel* isometric (WWᵀ = I); PCA orthogonality is orthogonality of the *components in the data*. An isometric kernel does not whiten Σ_out (Σ_out = W̃ Σ_patch W̃ᵀ keeps Σ_patch's spectrum up to compression), so the ONI network is "the width an orthogonally-constrained kernel yields", not "the width a PCA-optimal encoder would use". The three-widths sentence over-reaches in its first term. *Fix:* rename the first width "orthogonal-kernel width (D5)"; keep the point that forcing orthogonality raises the width and costs accuracy; drop the words "PCA-optimal".

**A5 (MAJOR). "The width the task needs" is not Γ's τ-width.** Γ at the end of training on a converged network is dominated by the remaining hard examples (the package's own P-b7.6: follows the EL2N > 0.1 fraction); its τ = 0.999 width (179–312 of 512) counts directions the *residual error* touches, not directions the task needs — under collapse the task needs K − 1. Cui et al.'s active-neuron count is a loss-sensitivity count, which is a different reading again. *Fix:* call Γ's width "the gradient-side width" and say what it measures (the span of the window's errors), not "needed by the task".

**A6 (MAJOR). ResNet-34 is said to keep the order and to lose it, two pages apart.** 3-D: "within role, 3×3 order is kept in VGG-16/ResNet-18/34" (public weights, timm reference, γ = 0); 3-D two lines later and P9.50: ResNet-34 under one recipe and γ = 1 reorders (+0.01/+0.09). Both are correct and both are stated; the reader sees a contradiction. *Fix:* one sentence that says the two numbers are the same finding — the reference draw decides the T–R of ResNet-34 (+0.61 at γ = 0, +0.13/+0.05 at γ = 1) — and then never again use "keeps/loses" for ResNet-34 without the reference.

**A7 (MAJOR; CONFIRMED 10-07 08:55 — their §2 assumes "a fully connected L-layer feedforward neural network", no normalisation layer anywhere in the paper; draft and package reworded to "does not transfer").** B6 may not remove anything from Zangrando et al. B6 shows the regularised objective has no stationary point with W ≠ 0 in a conv–BN layer. Whether Theorem 3.5 of Zangrando et al. is stated for networks *with* batch normalisation has to be checked in their paper; if their setting has no normalisation layers, B6 does not "remove the hypothesis", it says the setting differs, and the sentence "the rank-K bound has nothing to bite on" is an over-claim. *Fix:* quote their setting; if BN is excluded, write "their setting excludes the layers we measure".

**A8 (MAJOR). The backward chain is an inclusion loose by an order of magnitude plus an identity.** Σ_out ⊆ span(Γ_win) with k*(Γ_win) = 475 against 40 (full-width network); Σ_out = Γ^{1/2} M̃ Γ^{1/2} holds for any two PSD matrices on a common span. Calling this "the chain run backward" gives it the same status as the forward chain, which is exact for the linear step. *Fix:* in the preamble, label the two directions asymmetrically: forward = law (exact at the linear step), backward = inclusion + ordering + where the narrowing is measured.

**A9 (MAJOR). B9 is a tautology plus one setting.** Lemma B9.0: if the variance outside the span of P·K means is below 1 − τ, then k*_τ ≤ P·K − 1 — that is the definition of rank of a span. The empirical cliff at 4K (18/18, blind at K = 13, 20) is the real content, and it fails in the first extension (48 px). A referee will ask why a result that fails at 48 px is in the theory section. *Fix:* move B9 to the measurement section as "collapse is per position in VGG-16/CIFAR at 32 px"; keep the lemma as a one-line remark; drop "the one quantitative statement that holds" from the package.

**A10 (MAJOR). Attribution by exclusion ends at a non-answer.** "The data at this depth" is not a factor; a referee will ask which property of the data. A28/A31 excluded K, resolution, image count; what remains is image statistics × depth × steps, untested. *Fix:* either state the open factor list as the finding ("none of K, resolution, N restores the order; the remaining candidates are …") or run the one cheap contrast not run (same depth, same recipe, CIFAR-10 images upsampled to 224 in the ResNet-50 — the A7 surrogate says the spectrum is the causal channel at initialisation; after training it is untested). Not proposing it; naming it.

**A11 (MAJOR). Family-level claims rest on n = 1.** Only ResNet-50 has six recipes. Every "family" sentence (dense, inverted residual, transformer) is one checkpoint against one reference draw; the image bootstrap shows images are not the noise source, so there is no error bar at all. *Fix:* no sentence that ranks families; only within-family statements with the recipe count stated.

**A12 (MAJOR). The question promises prediction during training; the answer is "not derived".** §1 asks "how much of it can be … predicted during it". 3-C ends with "there is no derivation of κ". The title candidates all promise an answer. *Fix:* reword the question to what is delivered: computable before training, bounded and ordered during it, located (M̃) but not derived at the end.

**A13 (MINOR). Σ_patch and K-FAC's A factor may differ by centring.** K-FAC's A = E[aaᵀ] is uncentred; whether the paper's Σ_patch is centred is not stated in the package. Γ is uncentred (gradgram accumulates the second moment). *Fix:* state both conventions; "the K-FAC factors up to centring" if needed.

**A14 (MINOR). "Late τ-width 20–50 channels for K = 10" and "k*(0.999) ≤ 40" are two thresholds without saying so.** *Fix:* give τ with every width.

**A15 (MINOR). Transformers: "preserves" is reference-dependent (A33: +0.52/+0.48 against the faithful draw, fc2 negative).** The package says it; the family sentence in 3-D does not. Same fix as A6.

**A16 (MINOR). The reproduction gate for Yu & Wu is fc1 only; qkv is 0.746× their level.** Fine as stated; do not write "reproduced" without "fc1".

**A17 (MINOR). The demoted method half.** The paper is neither a method paper (controller negative in 3/4 cells, operator known) nor a theory paper (A1). It is a measurement paper. TPAMI accepts those when the measurement changes how the field reads its own results (Network Dissection) — here M1 (r_max reverses 14/14) and "collapse theory does not describe the position-sampled conv width" are the candidates for that role, and they are in §2 and 3-C, not in the title.

**A18 (MINOR). 162 predictions with 47–53 failures reported is a strength only if the failures are visible in the main text.** At present they are in Table S17. *Fix:* one paragraph in the main text listing the failed predictions that changed a claim.

**A19 (MINOR). Cui et al. 2020 is now the prior definition of the gradient-side width.** Any sentence saying "we introduce the gradient-side count" must go; B7's k*(Γ) is "Cui's count at channel level".

## B. The five questions

1. *Big enough; gap real?* The question is big; the delivered answer is half. The gap (no line states the layer-wise profile or computes it before training) is real after the Cui/K-FAC/FWSVD check — none of them does the per-layer profile with random references.
2. *Strike / over-stated / known.* Strike: "the cap sets the order" (A2), "PCA-optimal width" (A4), "task needs" (A5), "B6 removes Theorem 3.5" until checked (A7). Known and must be framed as such: scale invariance (B6), memory window (B7.1), the Ky Fan instances (A1), the gradient-side count (A19).
3. *Theory contribution or review with measurements?* A review with measurements, plus the measured hypothesis failures. That is a legitimate TPAMI contribution if the paper says so; it is a rejection ground if the paper calls it theory.
4. *Most damaging objection.* "A careful measurement study whose theory is a reading list and whose one mechanism (what sets κ) is not delivered; the net-effect claim is attribution by exclusion." Cheapest surviving framing: lead with what the width is *not* set by — the channel count (M1), the collapse theorems (3-C, B6, A35), the forward data after training (B7 sweep) — and the one thing it is computable from (the initialisation chain); title in that key.
5. Scores below.

## C. Scores (1–5) and probabilities

| Axis | Score | Why |
|---|---|---|
| Literature | 4 | six lines mapped, nearest neighbours read in full, three new boundaries today; remaining: Zangrando's setting (A7), Cui citations, Google Scholar rescan |
| Size of the question | 3.5 | big question, half answer; the promise in the title exceeds the delivery (A12) |
| Theory | 3 | correct application of known results with measured hypothesis failure; no theorem of substance; the spine clarifies but does not add |
| Data | 4 | 44 checkpoints, controlled ImageNet trainings, 162 registered predictions; n = 1 outside ResNet-50 |
| Usable tool | 3 | the protocol package; the method half is negative |

Acceptance as submitted (after the pending text corrections, framed as the package frames it): 30 %. With the fixes above (A1, A2, A4–A6, A8, A9 are rewording; A3 is a script; A7 is a check): 40–45 %, with major revision the likely first decision (50 %).

## D. What I would do before the rewrite, in order

1. A3 (script the ledger summary) and A7 (read Zangrando's setting) — hours, no money.
2. Reword per A1, A2, A4, A5, A6, A8, A9, A12 in the draft and the package — half a day.
3. Then the rewrite under the measurement-paper framing, with the theory section titled for what it is.
