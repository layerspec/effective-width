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

## D. 其他今天判定帶來的

- 控制器：P-l1 不成立 → 「shrink-only water-filling」，放運算子降為技術細節；P-v2 不成立 → ViT 上不得寫控制器優於均勻。
- 「殘差流保秩」：只對絕對譜成立（P9.60 不成立、P9.61 成立）；A27 → plain 對殘差的二分法撤回。
- A17：`sec:learned` 的「data or recipe」改「the data at this depth」，配方與 block 型別排除。
