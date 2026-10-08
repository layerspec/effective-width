# 選項 1（量測論文）骨架——2026-10-08

**用途.** 三週賭注（到 10-28）若無定理，照這份骨架投；賭注中的改寫（pending §A–H）也從這份做，整篇重寫、不 patch（作者 10-02 規則）。
**來源.** 框架審稿 10-07 的存活清單（`radar-2026-09-30.md` 末節）、`hostile-review-agent-2026-10-07-response.md` §D／§E、`claim-boundaries-2026-10-01.md` 14 條、`pending-text-corrections-2026-09-30.md` §A–H。
**不變的約束.** 12 頁含參考文獻（Pagella 量頁）；專家讀者、每段菁華；每個數字有檔；T–R 全退場（H1）；任何隨機對照寫明初始化（A33）；探索性標示為探索性。

## 一句話

一層卷積輸出的位置取樣共變異數、達 τ 變異的主成分數，除以可達秩：量它需要四個門檻；量了 44 組權重、11 個 CNN 家族與兩個 ViT 家族之後，初始化的剖面可由架構＋影像二階統計模擬，訓練把水準抬高、在深 ResNet 的第三 stage 把格內的下降翻成上升、在少類別任務把末段塌縮；讀出的寬度在少類別可零重訓摺入。

## 標題

作者偏好候選三的形式，但副標題「Initialisation Sets the Order」已站不住（§10.35）。可用的副標題陳述只有「水準」與「格內趨勢」；是否拿掉 *convolutional*（A30）由作者定。本骨架不定標題。

## 章節與每節承重的主張（括號＝證據檔；★＝這次重寫要動的）

**I. Introduction（1.5 頁）**
- 問題句：一層用了多少寬度、兩份已發表的剖面為何不同（Garg 2024、E&B 2024）。
- 貢獻四條：(1) 協定與四個門檻（n/C、r_max、位置 vs 池化、conv vs block），每個門檻移動剖面的量 > 文獻報的效果；(2) 44＋ 組權重對自己的初始化：初始化剖面可由架構＋影像二階統計**模擬**（層內律 535 層；替身 ≥ 3 種子才寫 simulable，H16）、訓練抬水準、第三 stage 格內趨勢翻轉（26/26、22/24）、少類別末段塌縮（A24／A25／A27／A31）；(3) 分解 Σ_out ＝ W Σ_patch Wᵀ：訓練把瓶頸從資料移到核、核增益部分由 NFA 解釋（A32，Beaglehole 2024／CRH 劃界）、ONI 反事實（H8 措辭）；(4) 讀出的寬度：少類別零重訓摺入、重訓、一次訓練＝均勻（尺讀滿時）。
- ★Related Work 依六線各一段「停在哪、空格在哪」＋ 14 條劃界句（claim-boundaries）；最近鄰 Harun／Masarczyk（問同一組因素）、Garg、Yu & Wu、NSC、Spectral Lens（H11）。

**II. Method（2 頁，大致不動）**
- 對象（位置取樣 conv 輸出共變異數）、統計量（k*_τ、PR；四種維度分開，規則 4）、估計、樣本門檻（H15：門檻依據是經驗平坦化 Fig. S1）、模型與對照——★對照表加初始化欄（γ＝0／γ＝1／JAX／LayerScale；A33）。

**III. Results（6 頁）**
- A 對已發表剖面的驗證（Garg r＝0.98；E&B 7 個過門檻 block）。B 可達秩（M1；H12 措辭：apply Garg 的正規化）。C 跨 checkpoint 的變異（ResNet-50 六配方 SD 0.05）。D 池化 vs 位置（687/703）。E conv vs block（161/161）。F 統計量一致性。G depthwise vs dense。——以上不動，只核數字。
- ★H **Trained versus initialised**：重寫。(i) 水準：所有家族 R→T 上升（表 families，T–T 可報的家族寫 T–T，n＝1 家族不排序，H9／H17）；(ii) 格內趨勢：第三 stage 3×3 寬度訓練後隨 block 序上升（公開 26/26、受控 22/24、CIFAR bottleneck 3/3），γ＝1 初始化同格下降、γ＝0 無序，「順序丟失」＝兩者之差（§10.35）；初始化下降的機制寫一段（§10.36–10.45：加法步、分支窄於流、白噪重現；已知映射鏈，無定理）；(iii) 深度／資料／配方的受控跑（A17、A28、A29b、A31）只報格內趨勢與水準，不寫歸因（H2）；(iv) transformer 同一把尺（A30；兩種參照，H14）；(v) 少類別塌縮（A24／A25／A27／A26；H10 措辭：步數主導、影像數反向且較小）。
- ★I **Decomposition**：Σ_patch／核增益；A32（共用列空間的底 0.8、偏相關 +0.5、「部分由 NFA 解釋」）；ONI（「input-side width unchanged by ONI」）。
- ★J **Widths read from the covariance, used three ways**：摺入（Chee／Yu & Wu 同運算子、劃界）；重訓（CIFAR-10 倖存、ImageNet 空測試）；一次訓練一段（「尺讀滿 ⇔ 控制器＝均勻」，P-l1／P-l2–l4／P-v2 全寫）；B9 格上限只當 32 px 的探索性觀察（H5）。
- ★K **What the published theorems say about this object, and where their hypotheses fail**（≤ 0.75 頁；H3／H7）：層內律（S6，精確）；塌縮定理的對象是池化後特徵（未被檢驗，A34／A35）；B6 一句 remark＋四引用、Zangrando 設定不適用；B7 包含關係＋順序（H6）；精確陳述全進補充。

**IV. Discussion（1.5 頁）**：報告要求（每個效果配自身抖動／CI）；寬度選擇的含意（少類別；多類別無閒置，判準 τ＝0.999 不重訓）；★Limitations（H15：不量深度軸、stem 與資料集混淆、單一 checkpoint 的家族、控制器四格三負、B9 範圍）。

**V. Conclusion（0.3 頁）＋ Reproducibility（網址待填）。**

**補充（現 21 頁，重排）**：S1–S19 表與圖；理論陳述（S2–S7、B6、B7、B9.0 引理、B11.0）；S17 預登記表加 blind／post hoc 欄，總數只引 `prereg_counts.py`（H13）；搜尋紀錄一節（第 1 軸 4.5 條件 (d)）；bootstrap／效果量表。

## pending 清單對章節的映射

| 章節 | pending 項 |
|---|---|
| I | C（文獻）、F 的劃界條、H3、H11、H12、H13 |
| II | A（五個權重更正）、B（對照初始化）、H9、H15 |
| III-H | A、B、D、E、G、H1、H2、H10、H14、H16、H17 |
| III-I | C 末條（CRH／Beaglehole）、H8 |
| III-J | G（P-l）、H5、H6 |
| III-K | H3、H6、H7 |
| IV | E（A26／A27 撤回句）、H15 |

## 做法與順序

1. 作者看過這份骨架（標題、要不要 III-K 這一節、transformer 進不進正文）。
2. 先重產所有表與 `checkpoint_analysis.txt`（A33 初始化欄要加進 `table_families.tex`）。
3. 依章節順序整節重寫，每節寫完對照本骨架與 pending 映射；全文寫完做一次通讀（去掉修訂痕跡）。
4. 英文定稿後給作者中文稿（memory `chinese-manuscript-for-author-review`）。
5. 投稿前：bib 核對慣例、2026 新作重查、`\todo` 網址。
