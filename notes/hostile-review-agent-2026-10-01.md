# Hostile referee report (agent), 2026-10-01

Scope: `paper/main.tex` (12-page manuscript), `paper/supp.tex` and the `paper/table_*.tex` it inputs; the planned reframing (`notes/topic-statement-2026-09-30.md`, `notes/theory-chain-draft-2026-09-30.md`, `notes/claim-boundaries-2026-10-01.md`); the evidence base (`results/checkpoint_analysis.txt`, `notes/analysis-plan.md` §10, `notes/preregistration-ledger-2026-09-25.md`); the proofs (`notes/proofs-2026-09-29.md`). Line numbers are those of the files on 2026-10-01. Numbers marked "recomputed" were recomputed from the tables or from `checkpoint_analysis.txt`.

## Verdict

The manuscript is a careful measurement study whose headline interpretive claims no longer match its own evidence base. Three structural problems dominate. (1) The trained-versus-initialised comparison, which carries contributions 3 and 4, the abstract and the conclusion, is a property of the *pair* (profile, reference draw), and Table I silently mixes two different reference initialisations across its rows (torchvision γ=1 for ResNet-50, timm γ=0 for ResNet-18/34/101/WRN/ResNeXt); under one reference the "bottleneck gradient" and the "depth is not the cause" inference reverse, and the paper's own later experiments (A17, A29, A33) show a basic-block network reordering on ImageNet. (2) Several paragraphs of §III-H(iii), §III-I, §III-J and the Limitations still state numbers that the regenerated tables contradict (MobileNetV3/EfficientNet T–T −0.18/0.40 vs +0.88/+0.93; "the experiment ... is left to a study with the compute" when it has been run; "whether it is a property of plain networks is the registered question" when it has been judged). (3) The theory is thin and partly mis-attributed: Proposition S6 is the definition of a Wishart-type matrix, Proposition S7 is content-free, the alignment result is cited to Saxe et al. whose model assumes whitened inputs, and the attribution of the alignment to "input principal directions" is now known (A32) to be mostly explained by the output-gradient structure. The protocol work (r_max, pooling, block output, gate, validation gates) is sound and publishable; the paper around it over-reaches. Estimated TPAMI acceptance as it stands: 20–30% (major revision at best; a referee who checks Table I against Table S9/plan §10.16 rejects). The planned reframing ("what sets the width of a layer") answers the question only at initialisation; the trained level, the late-layer cap κ and the ImageNet reordering are measured, not explained, and the theory links are either known results or vacuous bounds (B2 closed, Th2/Th5 known, B7 pending). As planned: 25–35%, rising to 45–55% only if κ is derived or the reordering is given a tested mechanism.

## Findings (most severe first)

1. [B/A] [fatal] Table I mixes two reference initialisations; cross-family inferences built on it are invalid.
   Location: `paper/table_families.tex`; `main.tex` 721–737 ("Depth is not the cause, since ResNet-101 agrees with its initialisation more (0.20) and Wide-ResNet-50-2 and ResNeXt-50 more still (0.33, 0.46), the four bottleneck families ordering as a gradient"); supp claims table 697 ("Bottleneck families learn their ordering, as a gradient ... ResNet-101 above ResNet-50"). Evidence: plan §10.15 (c), §10.16(B), §10.22: the ResNet-50 random control is torchvision-built (γ=1 in the last BN); the ResNet-18/34/101/WRN/ResNeXt controls are timm-built (γ=0, zero-init residual). Under γ=0 ResNet-50 T–R is +0.44 (not −0.08); under γ=1 ResNet-101 is −0.59 (not +0.20), ResNet-34 +0.13 (not +0.61), WRN −0.24 (not +0.33). Under one reference the depth ordering (R50 −0.08 → R101 −0.59; A29 20-conv +0.69 → A17 52-conv −0.11) is exactly what the text rejects. The row-level numbers are not wrong; the table is not comparable across rows and the text draws cross-row conclusions. Resolve: one reference per table (or both columns), state the γ convention in the caption, and rewrite (ii) and the claims-table row.

2. [A] [fatal] "measured against their own initialisations" (abstract 65–66, §I 138–140, §VI) is false for the timm-named families.
   The tv_in1k weights were trained from torchvision's γ=1 initialisation; their random controls are timm's γ=0 draws. A33 (plan §10.22, P9.69) shows T–R is a property of the control, not of "faithfulness". The abstract's central sentence rests on a mis-stated control. Resolve: "against a random draw of the same architecture (library default, γ as stated)".

3. [B] [fatal] §III-H (iii) states inverted-residual numbers the regenerated Table I contradicts.
   `main.tex` 739–746: "MobileNetV3-Large's four checkpoints share no profile (T–T −0.18, R–R 0.75) and EfficientNet-B0's five agree moderately (0.40, R–R 0.97); the 1×1 levels span 0.18–0.65 and 0.10–0.73". `table_families.tex` rows: MobileNetV3-L T–T +0.88, T 0.65; EfficientNet-B0 T–T +0.93, T 0.71. plan §10.16(A): the earlier numbers came from five weights fed the wrong input normalisation (top-1 2–46% at measurement). Same stale claim at 923 ("whose public checkpoints share no profile"), 1221–1222 (Limitations), supp claims table 701 (which already carries the corrected numbers, so supplement and main text contradict each other). Resolve: rewrite (iii) as "share a profile across recipes (+0.88/+0.93)".

4. [A/B] [major] "training halves the depthwise layers" (`main.tex` 743) is contradicted by Table I.
   Depthwise T/R: MobileNetV2 0.42/0.39 (rises), MobileNetV3-L 0.24/0.41 (×0.59), EfficientNet-B0 0.25/0.35 (×0.71). Recomputed from `table_families.tex`. Resolve: report the three ratios.

5. [A/H] [major] Contribution 3 / abstract / §VI "plain, basic-residual and dense families keep the architecture's ordering" is contradicted by the evidence base.
   A17 (plan §10.14): a 52-convolution basic-block network under one ImageNet recipe has same-seed T–R −0.21/−0.02 (bottleneck −0.25/−0.26); A33: ResNet-34 against γ=1 is +0.13. The "block type" dichotomy in §III-H (i)–(ii) does not survive; what the data show is a depth × dataset × optimisation-length effect (A28: K100-64 +0.02 at 37k steps → −0.32 at 366k). Resolve: replace the family taxonomy by the depth/length statement with the reference named.

6. [B] [major] Limitations says the decisive experiment was not run; it was.
   `main.tex` 1216–1219: "the experiment that would, the same architectures under one ImageNet recipe with seeds, is left to a study with the compute". A17 ran exactly that (basic [6,6,6,6] vs ResNet-50, same ImageNet recipe, two seeds; plan §10.14; H9.4a/b both fail). Likewise 1085–1086 "whether it is a property of plain networks is the registered question of plan Section 9.16" — A27 has been judged (P9.39/P9.41 fail, P9.40 holds; plan §10.18). Resolve: report A17/A27 and delete the deferrals.

7. [A] [major] Alignment attributed to "its input's principal directions" and to Saxe et al. is (a) mostly explained by output-gradient structure and (b) mis-cited.
   `main.tex` 66–69 (abstract "whose gain aligns with its input's principal directions"), 236–240 (contribution 4 "the deep-linear prediction of Saxe et al."), 927–931. A32 (plan §10.21): Spearman(s², output-gradient energy γ) = +0.98/+0.95; partial correlation of s² with input variance given γ is +0.55/+0.45 (registered ≥0.7 fails); the raw ρ_align 0.8–0.96 is more than half explained by γ. Saxe, McClelland & Ganguli (ICLR 2014) assume whitened input (Σ¹¹ = I); their result concerns the modes of the input–output correlation Σ³¹, not alignment of a layer's gain with its *input covariance*. The nearest published statements are Maennel et al. (NeurIPS 2020, first layer), Beaglehole et al. 2023 / Radhakrishnan et al. (Science 2024) (CNFA, every layer) and Beaglehole–Mitliagkas–Agarwala (TMLR 2024, NFA as alignment to the PTK) — none cited in `main.tex`. Resolve: cite the NFA line, report the partial correlation, drop "the deep-linear prediction of Saxe".

8. [F/A] [major] "no profile measured on trained networks divides by it" (`main.tex` 156) is false.
   Garg et al. 2024 (arXiv 2407.04797, "utilized rank") divide the per-layer rank of trained networks by min(m, d), i.e. the same attainable rank, and report per-layer values for ImageNet ResNet-50 and ViT-B/16 (`notes/a30-prior-art-2026-09-30.md`, claim-boundary #1). Yu & Wu (AAAI 2023) read per-block DeiT-B spectra and fold with the same PCA truncation. Neither is cited in `main.tex` or `refs.bib` as used there. The r_max contribution must be re-worded as "the activation-spectrum literature normalised by C ignores a known bound" and both papers cited. (Section III-A reproducing Yu & Wu's fc1 numbers, P9.67, is a validation gate the transformer section should carry.)

9. [A/B] [major] "merge to within the between-weights scatter" (§III-C heading claim, 499–511; claims table 683) is not what Table S14 shows.
   Equivalence within 0.05 holds in 7/14 models; the median residual shift −0.018 has a 95% CI excluding zero; A (probability of superiority) is 0.11–0.67. The populations are *closer*, not equivalent, in half the models. `notes/claim-boundaries-2026-10-01.md` #1 proposes the sentence "closes to within 0.05 in 14 of 14 models", which contradicts the table outright. Resolve: "within 0.10 in 12/14, within 0.05 in 7/14; residual deficit −0.02".

10. [B] [major] "a trained expansion reaches 94–99% of its bound" (`main.tex` 507–509) contradicts Table S8 three lines earlier.
    Table S8: attainable ratio 0.990 (1/6), 0.938 (1/4), **0.820** (1/2). The ratio-2 layers reach 82%, below the unbounded layers' 94–97%. The claims-table row "Trained expansion layers are nearly full-rank ... 0.94 (ratio 4), 0.99 (ratio 6)" omits the 0.82 row. Also Table S8's layer counts sum to 249 while the text says 261 bounded layers (recomputed from `table_models.tex`: 1+20+3+1+3+5·20+37+21+21+2+18+15+17+1+1 = 261); 12 layers are unaccounted for and the "1/4 = 141" row cannot be reconstructed from 17 quarter-bound layers per ResNet-50 (16 conv3 + layer1 downsample; the other three downsamples are 1/2) × 6 + 34 for ResNet-101 = 136. Resolve: reconcile the counts, report the 0.82 group.

11. [B] [major] Three different tallies of registered predictions.
    `main.tex` 1203 "Fourteen of the registered predictions ... failed"; `supp.tex` 667–669 "21 held, 17 failed, 4 reversed, 1 by letter, 25 pending" (dated 09-25); ledger "統計" 22/18/4/1/3/2/1/1/22/3 with a note that counts stopped on 09-28; the ledger itself now has ~110 rows (A28, A30, A32–A35, B7 added) and `table_prereg.tex` says "77 rows". Resolve: one generated tally, dated, consistent across the three places.

12. [B] [major] Supplementary claims table reports the E&B agreement over the 16 ungated blocks as the gate evidence.
    `supp.tex` 680: "16 ResNet-50 blocks, ρ = 0.997, median |Δlog| = 0.063 ... gate (expl.)". `main.tex` 460–470 says the 16-block figure "is not evidence" and reports 7 gated blocks, median 0.035, max 0.098, ρ 0.96 (recomputed from `checkpoint_analysis.txt` 19–35: ratios 1.033, 1.036, 1.036, 0.973, 1.103, 1.092, 1.088; median |log| 0.035; one rank swap → ρ 0.964). Also unreported: ours is higher on 15/16 blocks (median ratio 1.065) — a systematic +6.5% bias, not "agreement". Resolve: claims-table row → 7 blocks, and state the bias.

13. [B/G] [major] Supplementary table numbers cited in the main text are off by 1–2.
    Float order in `supp.tex` (labels at 59…683): S14 = equivalence, **S15 = tab:sa18**, S16 = tab:sfreeprob, S17 = tab:sfreeprobsum, S18 = tab:claims, S19 = tab:sprereg. `main.tex` cites "Table S16" for the retraining table (986, 1063), "Table S17" for the pre-registration table (1207) and "Tables S18, S19" for the law tables (776). Hard-coded numbers instead of \ref. Resolve: use xr \ref or regenerate.

14. [A/B] [major] "within one channel at every layer" (`main.tex` 424–427; `supp.tex` 152–156) is contradicted by Table S4.
    `table_ablation.tex`: max |Δ| in units of r_max is 0.016 (4 positions), 0.016 (resize), 0.016 (COCO) and **0.125** (BN output). At r_max = 64 (the narrowest ResNet-50 layers) 0.125 is 8 channels; at r_max = 512 it is 64; even 0.016 is 8 channels at r_max = 512. The "4 positions" row also covers only 42 of 53 layers (the wide layers fail the gate at p = 4). Resolve: quote the max |Δ| and the layer count.

15. [D] [major] The sample-size gate is derived for independent observations; the 16 positions per image are dependent, and the text calls Np the "conservative count" (346), which is backwards.
    Proposition 1 and the "4% at n/C = 50" figure use n = Np. On the 6,400-image subset the widest layers sit exactly at n/C = 50 with n = 102,400 dependent samples from 6,400 images; the effective n lies between N and Np (`notes/proofs-2026-09-29.md` Th2 (iii) says so; the paper does not). The empirical flattening curve (Fig. S1) is the real defence and should be stated as such. Resolve: say "n = Np overstates the independent count; the gate is calibrated empirically (Fig. S1)".

16. [E] [major] Proposition 1 as stated is not correct "to first order in 1/n".
    `main.tex` 357–366: E F̂(k) − F(k) = (1/(n tr Σ)) Σ_{i≤k} Σ_{j>k} λ_iλ_j/(λ_i − λ_j). This is E S_k / E T̂ − F(k); F̂ is a ratio, and E[S_k/T̂] carries O(1/n) terms −Cov(S_k,T̂)/t² + Var(T̂)E S_k/t³ of the same order (Var T̂ = 2 tr Σ²/n). They are small relative to the bound (ratio ≈ 1/(k·PR)) but the displayed equality is wrong at the order claimed, and the proof ("the trace is unbiased") does not address the ratio. The derivation also assumes Gaussian data (Lawley) — never stated. Resolve: state as an inequality for E S_k/E T̂ with Gaussian assumption, or add the delta-method terms.

17. [E] [major] Proposition S5 (water-filling) is proved for the wrong objective on the shrink side.
    `supp.tex` 441–447 and proof 503: D_ℓ on the shrink side is tr R_S / tr Σ_a, the column-subset residual; the proof uses D_ℓ(C) − D_ℓ(C+1) = λ_{C+1}/tr Σ, the eigenvalue tail. The CSSP residual is not in general convex in |S|, and R_S for greedy S is not the tail. ω_ℓ is undefined. "Greedy discrete exchange ... terminates" proves nothing about optimality. Resolve: state the proposition for the spectral surrogate and say the controller uses it as a surrogate.

18. [E/F] [major] Proposition S3's bound for greedy selection mis-cites Çivril & Magdon-Ismail.
    `supp.tex` 497: "volume sampling gives the factor m+1 [deshpande2006volume], greedy selection a comparable factor [civril2009greedy]". Deshpande et al. 2006 give E‖A − π_S A‖²_F ≤ (k+1)‖A − A_k‖²_F *in expectation*; the proposition states it deterministically "for a volume-sampled S". Çivril & Magdon-Ismail 2009 bound the greedy *volume* approximation; no (m+1)-type Frobenius bound for greedy CSSP exists (the known greedy bounds, e.g. Altschuler et al. ICML 2016, carry condition-number factors). Resolve: "in expectation"; drop the greedy claim or cite a correct bound.

19. [E] [major] Proposition S6 is a definition, not a result; the paper leans on it as theory.
    `supp.tex` 459–462: for Gaussian W, WΣWᵀ "has exactly the law of W̃ΣW̃ᵀ" — this is the definition of a (scaled) Wishart matrix with population covariance Σ and C samples; the second half is Silverstein 1995 verbatim. The 535-layer "verification" (Tables S16–S17) checks that a Monte-Carlo draw of the same Wishart model reproduces a Wishart draw. `main.tex` 768–776 presents this as "the width a convolution produces has an exact law", contribution-level. The genuinely open step (ReLU, correlated positions) is admitted to fail (supp 644–645). Also: PyTorch's default `Conv2d` initialiser is Kaiming-*uniform*, for which rotation invariance (hence "depends only on the spectrum") does not hold exactly; torchvision/timm override to `kaiming_normal_` for most but not all families. Resolve: downgrade S6 to a remark, state which families are Gaussian-initialised.

20. [E] [minor] Proposition S7 is content-free as stated.
    "depends on the images only through the Gram matrix of the first-layer patches G₁ over all positions and images" — G₁ determines X₁ up to a rotation that the Gaussian first layer absorbs; nothing is reduced (the Gram has ~10⁵–10⁶ rows). The proof also says BN "acts sample-wise", which is false in training mode (it uses batch statistics); the conclusion survives because BN is a function of the whole matrix. Resolve: delete or state the actual content (first-layer rotation invariance).

21. [H] [major] "the width separates into an architectural, a data and a kernel factor" (abstract 60–62) is an overstatement of Corollary 1.
    k*(τ; WΣWᵀ) is not a function of k* of the factors; the corollary brackets it by κ, and at the trained median κ = 10.6 the bracket is [k*(0.14), k*(0.9995)] at τ = 0.95 (supp 522–524) — "empty on about half the layers" by the paper's own Limitations (1193–1195). The decomposition is exact; the *separation of the width* is not. Resolve: "the covariance factorises; the width is bracketed, informatively on 246/465 layers".

22. [C/H] [major] All-layer T–R for DenseNet-121, Wide-ResNet-50-2 and ResNeXt-50 is carried by the role alternation the paper elsewhere denounces as a confound.
    `checkpoint_analysis.txt` 424–448: DenseNet-121 all +0.71 but 3×3 role −0.26 (ceiling 0.56) and 1×1 +0.39 with levels 0.78 vs 0.47; WRN all +0.33, 3×3 −0.17; ResNeXt all +0.46, 3×3 −0.22. The two-population structure (1×1 vs 3×3 levels) produces a positive all-layer rank correlation regardless of learning — exactly the sawtooth argument of §III-C. The within-stage control was applied to ResNet-50 only. Resolve: report within-role T–R for every family, or rank within role.

23. [C] [major] H9.1's Mann–Whitney p = 0.2 (`main.tex` 757) is computed on nine dependent pairs.
    The nine T–R values per architecture come from 3 trained × 3 random networks; they are not nine independent observations. The same applies to H9.4a (4 pairs from 2×2; plan §10.14 notes this but the manuscript does not). Resolve: report the all-above-all rule only, or a permutation test over networks.

24. [C] [major] The controller attribution ("the timing and direction of its actions add about 0.2 each", `main.tex` 1033–1037) is drawn from the parameter-budget regime the paper itself declares confounded by compute, and is contradicted by P-l1.
    The 93.89 / 93.88 / 94.06 arms are P2h arms at 1.22–1.37× the full-width MACs compared with P2 (94.11, parameter budget); the paper says two paragraphs earlier that "under a parameter budget accuracy tracks compute". Under the FLOPs budget the shrink-only arm equals the full controller (93.61 vs 93.60, P-l1 fails; ledger), so "direction adds 0.2" is false where the comparison is fair. Resolve: delete the attribution sentence or restate it under the FLOPs budget with P-l1.

25. [B] [major] Two different margins for the same comparison.
    `main.tex` 1030–1031: controller is "0.5 better [than folding] at 0.60–0.69"; Discussion 1155–1157: "0.2–0.3 better than folding below 0.7 of the compute". From Table 2: 93.74 − 93.28 = 0.46. Resolve: one number.

26. [G/B] [major] Table 2 omits the arm that the text compares against.
    `main.tex` 1024–1026: "reaches the accuracy of the retrained ruler arms at 0.88 and 0.60 MACs (93.84 against 93.86, 93.74 against 93.57)". The 0.60-MACs retrained arm (conv-output ruler, 5.09M params, five seeds 93.57; `checkpoint_analysis.txt` 1030) is not in `table_controller.tex`, which the caption says "puts the three beside each other". Table S15 lists it with three seeds (93.55). Resolve: add the row.

27. [B] [minor] Same arm, two values, two seed counts, one pre-registration label.
    `main.tex` 988–989: "93.86 against 93.82; P9.17" (five seeds, Table 2); Table S15 and the claims table (718): 93.84 vs 93.77, "+0.07" (three seeds, the registered judgement). Resolve: cite the three-seed judgement with the registration, and say the five-seed numbers are the extension.

28. [C] [major] Selective reporting of the zero-retraining fold on transformers.
    `main.tex` 1012–1014: "2–7.5% of ConvNeXt-T's and ViT-B/16's for at most 0.28". `notes/controller-criterion-2026-09-16.md` 196/212: DeiT-S loses **0.9** at 95.9% of parameters (τ = 0.999, same operator). The sentence names the two favourable networks and drops the third. Resolve: report DeiT-S.

29. [C] [major] "accuracy is unchanged in six of eight checkpoints" rests on a 3,200-image McNemar test that cannot detect the changes at issue.
    `supp.tex` 573–580: 0.3 points = 10 images of 3,200; the ResNet-18 change is −0.4 (70.2 → 69.8, Table S11), outside the quoted "−0.3 to +0.2" range, and p ≥ 0.15 is a power statement, not evidence of no change. Resolve: report CIs on the difference, correct the range to −0.4…+0.2, and phrase as "not detectable at n = 3,200".

30. [B] [minor] "median 2.2 (over 19 networks)" for the pooling drop-to-step ratio is the median over 16 networks.
    `main.tex` 219–221, 579–580; `checkpoint_analysis.txt` 251–252: inverted-residual models median 1.50 (n = 3), "all other trained models" 2.20 (n = 16). Median of the 19 r_max ratios in `table_pooling.tex` (recomputed): 2.13. Resolve: 2.1 over 19, or say 16.

31. [B] [minor] Decomposition checkpoint count: "on ten of them" (abstract 67) vs "ten checkpoints in Table S12" (872) vs "six of nine checkpoints" (901) vs claims table "nine checkpoints".
    Table S12 has nine trained ImageNet checkpoints plus one random draw; the abstract's "ten of [the 44]" is wrong on both counts (these are first-measurement public checkpoints, nine trained). Resolve: "nine".

32. [B] [minor] Statistic-agreement across recipes quoted under two normalisations as if one.
    `main.tex` 632–636: "ρ = 0.85 for k*(0.95)/r_max, 0.86 for PR/C and 0.89 for erank/C"; Table S3 "recipes" column: PR 0.73, erank 0.70 (per r_max). The /C agreement is inflated by the architectural sawtooth the paper itself diagnoses in §III-C. Resolve: report all three per r_max (0.85/0.73/0.70) and drop "each statistic gives a reproducible profile" or qualify it.

33. [D/G] [minor] The supplementary ViT-B/16 profile was measured with the wrong input normalisation and has not been regenerated.
    `supp.tex` 335–356 and `table_vit.tex` reproduce `checkpoint_analysis.txt` §21 (ImageNet mean/std on an augreg checkpoint trained with 0.5/0.5). P9.55 (plan §10.19): re-measurement gives ρ 0.991 and a level shift of 0.026 — small, but the table and the "doubling it" sentence should carry the corrected numbers, and A30 (11 weights, 3 seeds) supersedes this one-seed section entirely. Resolve: replace with the A30 table and state the normalisation.

34. [D] [minor] Random controls measured with ImageNet statistics, trained weights with their own.
    plan §10.19: "各權重用自己的輸入統計量，隨機對照用 ImageNet 的". After the normalisation fix the T–R comparison pairs a trained profile on (0.5/0.5)-normalised inputs with a random profile on ImageNet-normalised inputs. Resolve: state it; re-measure the controls with the matching statistics for the affected families.

35. [A] [minor] "20 of the 53 convolutions are 1×1 expansions with r_max/C = 0.25" (`main.tex` 490–491, 211–213).
    Seventeen have r_max/C = 1/4 (16 conv3 + layer1.0.downsample); layer2/3/4.0.downsample have 1/2. Twenty is the count of r_max < C. Resolve: "20 bounded layers, 17 at a quarter".

36. [H] [minor] "Three tests decide. The bound is never exceeded" (492–494) is a tautology presented as a test.
    rank(WΣWᵀ) ≤ r_max holds by construction; max k*(0.999)/r_max ≤ 1 can only fail if r_max is computed wrongly (the sanity check of CLAUDE rule 3). It is a pipeline check, not evidence for the denominator. Resolve: move to Methods as a check.

37. [C] [minor] SO-versus-SRIP "ordering of the gains" rests on a rank correlation over 8 layers for VGG-16.
    `main.tex` 951–955; Table S10 L = 8 (five of thirteen VGG layers fail n/d ≥ 50). ρ = 0.91 over 8 points has a 95% CI roughly [0.55, 0.98]; the SRIP −0.19 over 8 points is indistinguishable from 0. The "form in which P9.4 holds" is the post-hoc reading the ledger flags (⚠ P9.4: `checkpoint_analysis.txt` judges the literal threshold as failed). Resolve: report CIs, label post hoc.

38. [C] [minor] P9.5 "uncorrelated with the width gained" (`main.tex` 956–957): ρ = 0.15 over 12 runs is uninformative; the registered |ρ| < 0.5 threshold is nearly unfalsifiable at n = 12.

39. [C] [minor] Linear-probe comparison omits that the probe dimension differs per arm.
    `main.tex` 1051–1055: full 512, post-activation ruler 168, conv-output ruler 40, uniform 298 (plan §10.9). Part of the 0.6–9.5-point drops is dimension. The plan says "要註明"; the manuscript does not. Resolve: state it, or probe on a fixed-dimension projection.

40. [C] [minor] "every trained recipe rises (+0.56)" (`main.tex` 729) quotes a mean (`checkpoint_analysis.txt` 396 "rho(depth) trained mean +0.56") as if it were a universal per-recipe statement; per-recipe ρ_depth values are not in the analysis file for the subset.

41. [B] [minor] Two values for the ResNet-18 γ=1 reference in the evidence base.
    plan §10.15: level 0.44–0.48, R–R 0.71 (2 seeds); §10.16(B)/§10.22: 0.41, R–R 0.82 (3 seeds). `theory-chain-draft` III-A(4) quotes the first. Resolve: one set with the seed count.

42. [G] [minor] Table S7 lists "ResNet-50 tv V1" twice with no marker and puts ✓ in the "Untrained" column on ten rows while the caption says the column marks "the two random initialisations".
    Resolve: mark the backend duplicate; define the column.

43. [G] [minor] Positions, crops and the subset are under-specified.
    Not stated in `main.tex`: how the p = 16 positions are drawn (uniform? stratified? borders excluded?), the resize/crop sizes ("centre-cropped" only), how the 6,400-image subset was chosen, the retained-direction rule for ρ_align ("retained right singular directions", 912), the null-direction threshold for "numerically null kernel directions" (878; plan §10.3 uses 1e-4/1e-6/1e-8), and the CIFAR recipe beyond "100 epochs, one-cycle" (learning rate, batch, weight decay, augmentation). Resolve: a recipe table in the supplement.

44. [G] [minor] Supplement self-description is stale.
    `supp.tex` 42: "gives the two further propositions and every proof" — there are seven (S1–S7) plus the truncation proposition; 407: "Proposition prop:truncation above" — it is stated *below*, in Further Results, after its own proof (470–471). Prop S4 uses Σ_z with z undefined there.

45. [E] [minor] Proposition S2's hypothesis is false for post-ReLU activations as written.
    "each almost surely in the range of Σ_act" — a vector with nonzero mean lies a.s. in μ + range(Σ), not range(Σ); post-ReLU activations have nonzero mean. The conclusion holds for the centred vector; the statement and proof should say so.

46. [E] [minor] Proposition S4's "variance a wider site can admit" assumes the isometric counterfactual silently.
    "a site of width C passes at most the top C directions ... raises the variance the site can pass by at most Σ_{C<i≤C+Δ} λ_i(Σ_x)/tr Σ_x" — with gain (P ≠ I) the output variance is unbounded by the input spectrum; the bound is for Σ∘. Resolve: state κ = 1 or phrase as directions, not variance.

47. [E] [minor] Corollary 1's last clause is redundant and its proof's wording wrong.
    If κ = 1 and C_out ≥ d then W∘ has orthonormal columns and Σ∘ carries *all* d eigenvalues of Σ_patch; "truncated to r_max = d dimensions" truncates nothing and min(k*, r_max) = k*. Harmless but a referee will notice.

48. [F] [minor] Function-preserving growth and gradient-directed growth have named ancestors not cited.
    `main.tex` 1016–1022 (grow with zero outgoing weights; grow along unmet-variance directions): Net2Net (Chen, Goodfellow, Shlens 2016) for function-preserving widening; GradMax (Evci et al. ICLR 2022) and Firefly (Wu et al. NeurIPS 2020) for choosing the directions of new neurons; DMCP/BCNet/Yuan et al. 2023 for learned-vs-uniform widths (`notes/p2m-prior-art-2026-09-30.md` knows all three). Only CHEX and NORTH* are cited.

49. [F] [minor] Masarczyk et al.'s "within a few hundred steps" is reported in Related Work (152–154) without the paper's own contrary finding.
    A34 (plan §10.20): on the channel covariance the numerical rank *rises* in the first 25 steps and never collapses to K (68–138 channels); the §III-J claim "reached by optimising long enough" (1093) and the related-work sentence are left unreconciled; the object difference (flattened vs position-sampled) is the reconciliation and should be stated once.

50. [A] [minor] "the pooled value is lower at every layer by more than the step" (Discussion 1125–1127).
    Not at every layer (16 exceptions) and not for DenseNet-121 (ratio 0.50). The bootstrap on the 6,400-image subset (Table S13) gives ResNet-50 a drop-to-jitter ratio of 0.83 on its 15 gated layers. Resolve: "at 687/703 layers; by more than the step in 18/19 networks".

51. [A] [minor] "most families keep the ordering" (abstract 66–67).
    Under the paper's own Table I, 4 of 11 families do not (R50, R101, WRN, ResNeXt vs ceilings 0.85–0.94); under a single γ=1 reference ResNet-34 joins them (5/11). "Most" is reference-dependent.

52. [A] [minor] Contribution 1's "to within 10% at every block that passes the pooled gate": the maximum is 0.098 in log units = 10.3%.

53. [B] [minor] Limitations (1223–1226) calls the ViT result "one checkpoint and one seed"; A30 (plan §10.19) has 11 weights × 3 seeds and P9.71 shows the transformer "keeps its ordering" only against the default (not the faithful JAX/LayerScale) initialisation — the same reference dependence as finding 1. `notes/claim-boundaries` #13 ("both ViT families keep their ordering") omits this; its "r = 0.965" is the merged 24-point figure, fc1 alone is r = 1.000.

54. [H] [minor] "a network started at the computed widths gains nothing" (`main.tex` 798–800, Discussion 1147–1149) is an apriori-*controller* arm at 1.32–1.37× MACs (P-h1), not a network "started at the computed widths" trained plainly. Resolve: say what the arm is.

55. [I] [major] Scope mismatch between title/abstract and body.
    The title promises protocol, pitfalls and the initialisation comparison; the body spends §III-J on retraining, zero-retraining folding, a training-time width controller (Table 2) and class-count controls, and the supplement adds a transformer and a layer-local law. Three external reviews already said "measurement report, no theory". A referee will ask what the paper *is*. The planned reframing fixes the title but (see 56) not the chain.

56. [H] [major] The planned reframing's central question is answered only at initialisation.
    `notes/topic-statement` and `theory-chain-draft`: link A (initialisation) is a Wishart draw plus a Gaussian-field surrogate — empirical, expected, and admitted to have no closed form for the ReLU step; link B (alignment) is the NFA literature; link C removes a bound (B6) and finds the collapse object is the pooled representation, leaving κ "measured but not derived"; link D explains the ImageNet reordering as "assembled consequence" of a cap κ(K) that A28 shows does not move with K within seed noise (K100 vs K1000 at 224: −0.18 vs −0.26; seed differences 0.11–0.27) and that only optimisation length moves; link E is Weyl. A sceptical reader gets off at III-C: "what sets the width" after training is not stated. Candidate title 3 ("Initialisation Sets the Order, the Task Sets the Level") is contradicted by R50/A17/A28 (order not kept at depth on ImageNet) and by A33 (order relation depends on the reference draw).

57. [E] [minor] Theory-chain III-C conflates objects before admitting it.
    "What we measure is above K by a factor 2–50" compares the position-sampled conv-output τ-width with theorems about the input of the final linear layers (pooled features); the draft then concedes the collapse object is the pooled representation (wcv 0.07). The "factor 2–50" paragraph should not precede that concession as if it were a discrepancy with theory.

58. [E] [minor] Lemma B7.1 and Th2/Th5 are textbook.
    B7.1 is the impulse response of a linear time-varying recursion (the proofs note says so); Th2 is Shawe-Taylor 2005 / Blanchard 2007 and Th5 the same with C as sample count (the note's own correction). The draft calls B6 "ours and small" — correct — but the reframing's theory axis then contains no new theorem; the paper should not present a "theory section" built from these.

59. [I] [minor] Abstract and §I do not say that the registered shape hypotheses all failed and that the headline findings are exploratory.
    `main.tex` 251–256 does; the abstract presents only exploratory results ("most families keep", "training moving the bottleneck", "whose gain aligns") without the label. A pre-registration-aware referee will ask.

60. [G] [minor] `\todo` at 1266 and the claim "pip install layerspec" (1262) — no evidence the package is published; the repository is private.

61. [B] [minor] `notes/claim-boundaries` #2: "reproduces every measured layer at initialisation to within two channels" — Table S17 max |Δ| is 10 channels (ResNet-34, τ = 0.95), 8 (ResNet-50, DenseNet); the median is ≤ 2. Resolve: "median ≤ 2, 94–100% within 0.05 r_max".

62. [D] [minor] Equivalence margin 0.05 (§III-C) is the per-layer SD across recipes used as a margin on a Hodges–Lehmann *median shift*; the two are different scales, and the margin was chosen after the data (the 09-23 note introduces it). Label as post hoc.

63. [G] [minor] ConvNeXt depthwise p-values (668–672: 13/18 p = 0.013, 12/18 p = 0.21) are Wilcoxon (sign test gives 0.048/0.12); the test is not named in the sentence.

## Counts

By class: A 12, B 15, C 10, D 4, E 11, F 4, G 7, H 6, I 2 (several findings carry two letters; counted under the first).
By severity: fatal 3, major 27, minor 33.

## Ten most damaging

1. Table I mixes γ=0 and γ=1 reference draws; the bottleneck "gradient" and "depth is not the cause" reverse under one reference (finding 1).
2. "Measured against their own initialisations" is false for the timm-named families (2).
3. §III-H(iii) and the Limitations still carry the withdrawn inverted-residual numbers; main text and supplement contradict each other (3, 4).
4. "Basic-residual families keep the ordering" is contradicted by A17/A29/A33; the Limitations claims the decisive experiment was not run (5, 6).
5. Alignment attributed to input principal directions and to Saxe et al.; A32 shows it is mostly output-gradient structure; the NFA line is uncited (7).
6. "No profile divides by r_max" is false (Garg et al. 2024; Yu & Wu 2023) (8).
7. "Merge within the scatter" / "94–99% of the bound" contradicted by Tables S14 and S8; S8 counts do not reconcile (9, 10).
8. Proposition 1 wrong at its stated order and derived for independent Gaussian samples the protocol does not have; Np called "conservative" (15, 16).
9. Controller attribution drawn from the compute-confounded regime and contradicted by P-l1; fold result on DeiT-S omitted (24, 28).
10. The reframing's theory chain contains no new theorem and answers "what sets the width" only at initialisation; the ImageNet reordering remains unexplained (56, 58).
