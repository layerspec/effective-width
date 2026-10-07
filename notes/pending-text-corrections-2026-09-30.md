# 正文待改清單（2026-09-30；資料與表已更正，正文等框架敵意審稿過閘後一起改）

表與分析輸出已經是更正後的數字（`paper/table_families.tex`、`results/checkpoint_analysis.txt` §8、補充主張表一列）。
下面是 `paper/main.tex` 裡**現在與表不一致**的句子，行號是 09-30 的版本。改稿時逐條處理，處理完刪掉這份清單。

## A. 五個權重的輸入正規化更正（plan §10.16(A)；P9.56、P9.57 不成立）

| 行 | 現在寫的 | 更正後的事實 |
|---|---|---|
| 557–558 | depthwise 在 dense 之下，「at 0.20–0.42」 | 0.24–0.42（MobileNetV3-L 0.20→0.24、EfficientNet-B0 0.22→0.25、MobileNetV2 0.42 不變） |
| 739 | 小標「Inverted-residual checkpoints agree weakly or not at all」 | **撤回**。改為：inverted residual 的公開權重跨配方共享剖面 |
| 740–741 | MobileNetV3-L T–T −0.18、EfficientNet-B0 0.40 | T–T **+0.88**（0.79–0.94）、**+0.93**（0.84–0.99）；R–R 0.75、0.97 不變 |
| 742 | 1×1 水準 0.18–0.65 與 0.10–0.73 | 0.62–0.68 與 0.70–0.73 |
| 743 | 「training halves the depthwise layers」 | depthwise 水準 隨機 0.41→訓練後 0.24（MobileNetV3-L）、0.35→0.25（EfficientNet-B0）：降 0.6–0.7 倍，不是減半；MobileNetV2 0.39→0.42 不降 |
| 743–745 | 「These checkpoints differ in pre-training data … a statement about the checkpoints one can download」 | 刪。差異來自量測時的輸入統計量，不是配方 |
| 757–759 | 「…so an inverted-residual network under one recipe has a profile and the public disagreement is the recipes'」 | 後半句刪：公開權重之間沒有不一致。H9.2 本身（MobileNetV2 三種子 T–T 0.99）不變 |
| 922–923 | 對齊的例外是 inverted residual（+0.11 到 +0.61），「whose public checkpoints share no profile」 | 數字不變（對齊量的是 `.ra_in1k` 權重，正規化無誤）；**子句刪** |
| 1200–1201 | 「MobileNetV3-Large's random draws agree only at 0.75, so its reference is itself unstable」 | 保留（隨機對照沒有重量） |
| 1221–1222 | Limitations：「The inverted-residual finding is about the checkpoints one can download」 | 刪 |

新增要寫的：
- 兩個家族的 dense 層現在是 T–R **+0.65**（0.34–0.74）與 **+0.57**（0.47–0.67）、水準比隨機對照高 0.13／0.16——
  與 ViT（+0.69／+0.61）同一區，屬於「保留初始化順序並加水準」的那一群。三分法第 (iii) 類不存在了。
- Method 一句：每個權重用它訓練時的輸入統計量；用錯時五個權重的 top-1 是 2–46%（對的是 74–77%），剖面相關 −0.06 到 +0.94。
- §8 偏離紀錄已有一列；Table S17 的 P9.56、P9.57 已填。

不受影響（已查）：671–672 行的「sign reverses … 0.19 and 0.16 … 0.51 and 0.56」與 577–580 行的池化比值來自第二輪的 `.ra_in1k` 權重；
Table I、池化計數 687/703、對齊的 ρ_align、分解表都不含這五個權重。

## B. 隨機對照的初始化（plan §10.16(B)、§10.19；P9.58 不成立）——等 A33（plan §9.23）判完再定寫法

- 任何「保序／重排」的句子都要寫明對照的初始化方式；家族表 ResNet 系與 transformer 要加欄（A33 判完後決定是「兩種並列」還是「忠實的為主」）。
- 「plain／basic 的順序由架構決定（T–R 0.6–0.75）」：γ＝1 下只有 ResNet-18 成立（ResNet-34 +0.13）。

## C. 文獻（`notes/a30-prior-art-2026-09-30.md` §4、三份 09-29 的查核筆記）

- r_max：「文獻忽略了一個已知上界」要限定為「以 C 為分母的激活譜文獻」；Garg 2024 用 min(m, d)。
- 零重訓摺入：transformer 段的運算子是 Yu & Wu 2023 的；CNN 段是 Chee 2022 的（已寫）。
- 「ImageNet 沒有閒置寬度」一律加判準（τ＝0.999、不重訓），引 Garg 2024。
- 對齊：補引 Maennel 2020、Beaglehole 2023／Radhakrishnan 2024、Boix-Adserà 2025。
  **A32 已判（plan §10.21）**：核增益首先跟著輸出端梯度能量（Spearman +0.95／+0.98），控制它之後對輸入變異的對齊剩 +0.45／+0.55——
  `sec:decompose` 結論句、§I 貢獻第 3 條、摘要「whose gain aligns with its input's variance」改為「partly accounted for by the output-side gradient
  structure of the neural feature ansatz」；補充加 CNFA 逐元素相關的結構性底（置換 Γ 不變、隨機網路 0.8）。
  **10-01 補**：這句的理論出處是 Beaglehole–Mitliagkas–Agarwala（TMLR 2024）——NFA 等價於左奇異向量對輸出端梯度特徵的對齊；引它，並引 Beaglehole–Súkeník–Mondelli–Belkin 2024（AGOP ⇒ DNC）接到第 3 環。
- (iv) 閒置寬度（`notes/s3-prior-art-2026-09-30.md` §5）：開頭先引 Masarczyk 2023 與 Harun 2024——類別數、影像數、寬度、解析度的方向都已報；
  我們的是寬度方向的數字、固定 τ、劑量反應、少類別 × 224² 這個空格。**「要優化夠久才出現」在兩把尺並列的量測做完前不得照現在的樣子寫**
  （Masarczyk：深層的秩在 75 步內塌縮）。**A34 已判（plan §10.20）**：在通道共變異數上兩把尺同向、都隨步數收縮，塌到類別數不重現；A25 的句子改為
  「both rulers keep contracting with optimisation steps; the collapse to the class count is a statement about flattened representations」。A28 不寫「發現類別數／解析度是因素」。Discussion 加未解問題：ResNet 不論有無 skip，秩都高於 VGG。

## D. 其他今天判定帶來的

- 控制器：P-l1 不成立 → 「shrink-only water-filling」，放運算子降為技術細節；P-v2 不成立 → ViT 上不得寫控制器優於均勻。
- 「殘差流保秩」：只對絕對譜成立（P9.60 不成立、P9.61 成立）；A27 → plain 對殘差的二分法撤回。
- A17：`sec:learned` 的「data or recipe」改「the data at this depth」，配方與 block 型別排除。

## E. 10-01 判定帶來的（A28、A27 三種子、A35、B7）

- **A28（plan §10.23）**：`sec:learned` 的歸因句維持「the data at this depth」（登記第三支）。加一句效應量：一百類、64 像素、每類 130 張各自都不把
  ResNet-50 的順序拉回（同種子 T–R −0.09…−0.29，與 224 像素千類剖面 T–T 0.84–0.90）；**步數會**——一百類 × 64 像素在 37,000 步是中間區（+0.02），
  366,000 步重排（−0.32）。開頭先引 Harun 2024／Masarczyk 2023（方向已知），不寫「發現因素」；Limitations 加「增強未動」。
- **A27 三種子（plan §10.18）**：`sec:why` 的 "the terminal collapse is the task's" 刪。改寫為「on ten-class ImageNet the $\tau = 0.999$ ruler keeps
  0.52–0.62 of VGG-16's parameters (three seeds; 0.67 on CIFAR-10) and 0.95–0.97 of ResNet-18's; VGG-16's idle width is not confined to its last three
  convolutions (0.44–0.90)」，不附機制；種子間逐層差到 0.3，不寫逐層細節（規則 10）。
- **A35（plan §10.24）**：第 3 環加 B6 一段：conv–BN 層在訓練模式下 ⟨∇L₀, W⟩ ＝ 0（量到 |cos| ≤ 0.04、梯度是 λW 的 1.3–13 倍），rank-K 界沒有駐點可咬；
  引 Chen–Bu–Yan 2024（初始化 0.0003）並劃界，引 van Laarhoven 2017／Li & Arora 2020／Zhang–Wang–Xu 2019（**bib 未核**）。
  Results 一句「late weights far from rank K (tail 0.4–0.6)」；Discussion 一句「NC1 of the pooled features is nearly complete at the last convolution
  (within-class fraction 0.07 on CIFAR-10, 0.41 at 100 classes) while the position-sampled τ-width and the weight spectrum are not」。
- **B7（theory-chain 判定段）**：κ 跟輸出端梯度 Gram 的寬度走（後三層 Spearman +0.86）、Γ 隨步數縮（108 對 310）、頭部順序可前向算（ρ 1.00）
  但數目在全訓練 CIFAR-10 高估 1.6–2.1 倍 → 只寫順序；難樣本數與 K 分不開（偏相關 +0.58）→ "consistent with, not separated from, the class count"。

## F. 獨立審稿 agent（10-01；`hostile-review-agent-2026-10-01.md`，回應在 `…-response.md`）新增的 41 條

改稿時以 response 表的「行動」欄逐條處理；這裡只列會動**陳述**的，其餘是數字與表的對帳（#10–14、25–27、29–32、35、41–44、50、52、63）：
- **#1／#2／#51 對照不一致**：家族表加初始化欄、兩種對照並報；摘要「measured against their own initialisations」改「against a random draw of the same architecture (γ as stated)」；刪「bottleneck 家族的 gradient」與「depth is not the cause」。
- **#5／#6**：「basic 家族保序」改為深度 × 資料 × 步數的陳述（引 A17、A28、A33）；Limitations 刪「沒跑」兩句，報 A17／A27。
- **#7**：對齊改引 NFA 線（Beaglehole 2023／2024、Radhakrishnan 2024）；Saxe 只當深線性對照、寫明白化假設。
- **#9／#21**：「merge」改效果量（S14）；摘要「寬度分成三個因子」改「共變異數分解、寬度被夾擠」。
- **#15／#16／#17／#19／#20／#45–47（第 3 軸）**：Prop. 1 改成比值的正確陳述＋高斯假設；S5 改譜替身；S6 降 remark＋列高斯初始化的家族；S7 改第一層旋轉不變性＋修 BN 句；S2 置中；S4 κ＝1；Cor. 1 末句。
- **#18**：Deshpande 改「in expectation」；greedy 界查 Altschuler 2016 後決定留或刪。
- **#22**：家族表加角色內 T–R（DenseNet 3×3 −0.26、WRN −0.17、R101 −0.08）。
- **#23／#37–40／#62**：檢定與相關加 CI、標 post hoc、只報 all-above-all；探針維度註明。
- **#24／#28／#54**：刪控制器「0.2 each」；報 DeiT-S −0.9；先驗臂寫明是控制器臂。
- **#48**：補引 Net2Net、GradMax、Firefly。
- **#57／#59**：III-C 段序；摘要標探索性。


## G. 10-02 判定帶來的（P-l2–l4、P-b7.1′、P9.85／86；A29b seed 0 暫定）

- **控制器段（`sec:a18`，main.tex 約 1035、1068–1074、1168、1183 行）**：×4 寬 ResNet-20 加進「尺讀滿 ⇔ 控制器＝均勻」的清單——
  全寬 stage 1／2／3 讀 0.79／0.90／0.91；0.60 MACs 下控制器 95.02、均勻 95.07、全寬 95.26（P-l2、P-l3 不成立，P-l4 字面成立但均勻臂同樣不掉分）。
  「ResNet-20's last stage reads 0.93–0.99 at three widths」那句改為四個寬度並寫明加寬後多出來的空隙在前段。**不得寫「加寬後出現閒置寬度」。**
  B+ 版本裡控制器只留一段：一格正結果（VGG-16 CIFAR-10）＋三格負結果（ResNet-20、×4 寬 ResNet-20、CIFAR-10 ViT）。
- **第 3 環（理論鏈草稿 III-C 已改）**：B7.2′ 寫成包含關係＋順序，不寫等式、不寫「界是緊的」；表裡同列報尺寬臂（比值 0.93，兩邊貼 C）與全寬臂（0.08）。
- **第 1′ 環（草稿 III-A(3) 已改）**：不寫「恆等捷徑保輸入」；寫 plain／dense／inverted-residual 三家族 9/9 忘、ResNet-18 不忘、ResNet-50 忘，標探索性。
- **A29b**：seed 1 對收齊並由 `section_a29b` 判定前，正文不引 +0.75／+0.01。
- **CRH／PAH（`notes/crh-prior-art-2026-10-02.md`；10-02 補）**：`sec:decompose` 與第 2、3 環引 Ziyin–Chuang–Galanti–Poggio（ICLR 2025）。
  A22 的對齊＝他們的 backward RWA、A32 的「增益跟輸出端梯度能量」＝ forward GWA、P-b7.2＝ RGA——**都只在秩相關的意義下**；以矩陣比例的形式在位置解析的
  conv–BN 層不成立（B8，plan §10.27：共對角 0.2 對隨機基底 0.1；s² ∝ γ^{2–5}）。劃界句在該筆記 §4b。bib 條目待核（ICLR 2025 proceedings；另 Ziyin–Wang–Li–Wu NeurIPS 2024）。

## H. 10-07 框架審稿後的措辭修正（我的 19 條＋agent 48 條；核對在 `hostile-review-agent-2026-10-07-response.md`；改稿時逐條處理）

| # | 位置 | 現在的句子／用法 | 改成 |
|---|---|---|---|
| H1 | 摘要、§I、§III-D、標題候選 | 「初始化的順序在深 ImageNet 網路丟失」「Initialisation Sets the Order」、任何全層 T–R | **T–R 全退場**。順序軸只寫不需參照的格內趨勢：訓練後第三 stage 的 3×3 寬度隨 block 序上升（公開 26/26、受控 22/24、CIFAR bottleneck 3/3）；γ＝1 初始剖面同格下降、γ＝0 無序；「順序丟失」是兩者的差（plan §10.35） |
| H2 | §III-D、A17／A28／A29b／A31 各段 | 「歸因停在 the data at this depth」「深度是軸」「轉折在 20–36 層」「K 不恢復順序」 | 刪歸因語言；A17／A28／A31 改報格內趨勢（都上升）與水準；A29b 改「ResNet-18 初始化就上升；ResNet-34／50 的第三 stage 由下降翻成上升」；A31 K10 格退化（15／29 層 k*＝1），P9.81、P-b9.7 不當證據 |
| H3 | 理論節標題、§I 貢獻 | 「theory section」「one exact proposition」「removes the hypothesis of Theorem 3.5」 | 節名改「What the published theorems say about this object, and where their hypotheses fail」；B6 一句 remark 加四引用（van Laarhoven、Li & Arora、Zhang–Wang–Xu、Kunin）＋「Zangrando 的設定是全連接無正規化，不適用」；精確陳述全進補充 |
| H4 | 3-0 骨架 | 「one principle」「PCA-optimal width」「task needs」「Markov chain of covariances」「K-FAC factors」「cost 0.10–0.16」 | Ky Fan 縮成一句記號；「isometric-kernel width」；「gradient-side width（Cui 2020 的 active-neuron count 在通道上的版本）」；「Markov in the activation law；covariance recursion closed only for the linear step」；「analogous to the K-FAC factors（經驗 Fisher ≠ Fisher，Kunstner 2019）」；刪準確率代價（P-d5.3 ±0.5 成立） |
| H5 | B9 段 | 「predictions committed before the spectra were opened」「the one quantitative statement that holds」「Lemma」 | 「exploratory observation on VGG-16/CIFAR-10 at 32 px; blind only at K = 13/20 on the layers with a cliff」；引理改一句 remark |
| H6 | B7／B10 段 | 「bound」「chain link」「κ follows Γ (+0.86)」「late level falls with λ」 | 「inclusion and ordering, loose by an order of magnitude on full-width networks」；B10 進補充分解表；「+0.86 is consistent with, not separated from K and steps（fixed-K −0.05…−0.49）」；「weight decay must be present; above 5e-4 the level is insensitive」 |
| H7 | III-C 開頭 | 「what we measure is 2–50 K」「the position-sampled covariance does not collapse」 | 定案：對象不同（位置當樣本），塌縮定理**未被檢驗**；P 因子當作量到的關係報（4、49） |
| H8 | ONI 段 | 「counterfactual that hits」 | 「the input-side width is unchanged by ONI（Ostrowski identity holds by construction at erank 1）」；SO／SRIP 的一半保留為描述 |
| H9 | §II 協定、M4、家族表 | p 值（Wilcoxon over layers）、「every effect next to its jitter」、家族排序句 | 層間自相關：p 值降為輔助；T–R／λ 水準／B10 寬度都要抖動或 CI；家族層級句限於 ResNet-50 與受控訓練（n＝1 家族不排序） |
| H10 | A24／A25 句 | 「contraction deepens with steps, not with image count」「class count enters monotonically（κ）」 | 「步數主導、影像數反向且較小（P9.34 反向）」；「monotone」從 κ 拿掉，A24 只當 CIFAR 上量到的數 |
| H11 | §VI 最近鄰、gap 句 | 「none asks what sets the layer-wise profile」 | 「no one measures the τ-width of the position-sampled conv-output covariance against a random draw across families」；補 Harun／Masarczyk 問同組因素；K-FAC／Cui／FWSVD／Chen & Zhao／Zhang 的劃界句 |
| H12 | 貢獻 2、M1 | 「the nominal denominator understates…」當作發現 | 「we apply the attainable-rank normalisation of Garg et al. to the activation-spectrum literature, where it was not used」 |
| H13 | 預登記段、Table S17 | 「162／164 predictions」「pre-registered」於存量資料項 | 數字只引 `prereg_counts.py`；存量資料項寫「hypothesis written before the stored data were read」；non-blind／post hoc 列不計入 |
| H14 | Transformers | 「preserves」 | 寫明兩種參照（預設 +0.69／+0.61、忠實 +0.52／+0.48、fc2 負）；順序句同 H1 改趨勢 |
| H15 | Limitations | — | 加：不量深度軸（引 Chen & Zhao 2019、Zhang 2022）；有效樣本數的門檻依據是經驗平坦化（Fig. S1）不是 Prop. 1；只用 CIFAR stem／ImageNet stem 各一（stem 與資料集混淆） |
| H16 | 「computable before training」 | computable | 「simulable from the architecture and the image second-order statistics」（≥ 3 種子後） |
