# 對 10-07 獨立敵意審稿（`hostile-review-agent-2026-10-07.md`，48 條）的逐條核對與合併判定

核對原則：每條對原始紀錄（`checkpoint_analysis.txt`、plan §10、各 `epoch_*_layers.csv`）重算或查對；成立／部分／不成立；
我自己的審稿（`framework-review-own-2026-10-07.md`，先寫）重疊處註明。**判定前先自己算過的數字**：A17／A29b 的逐 stage、逐 role T–R（下表 A）、A31 K10 的平手比例、§6／§8 的 /C 對 /r_max 值、§31 的陳舊、Zangrando 的設定。

## A. 我自己重算的逐 stage／逐 role 同種子 T–R（τ＝0.95，/r_max，過門檻 dense conv；epoch 0 對終態）

| 跑 | 全層 | stage 1 | stage 2 | stage 3 | stage 4 | 1×1 | 3×3 |
|---|---|---|---|---|---|---|---|
| A17 ResNet-50 s0 | −0.25 | +0.86 (10) | +0.15 (13) | −0.40 (19) | +0.30 (10) | −0.49 (33) | −0.15 (16) |
| A17 ResNet-50 s1 | −0.26 | +0.87 | +0.43 | −0.15 | −0.15 | −0.55 | −0.28 |
| A17 basic6666 s0 | −0.21 | −0.33 (12) | +0.01 (13) | −0.22 (13) | −0.37 (13) | −0.13 (25) | −0.30 (24) |
| A17 basic6666 s1 | −0.02 | −0.14 | **−0.81** | **+0.72** | +0.44 | +0.19 | −0.12 |
| A29b ResNet-18 s0 | +0.75 | +0.80 (4) | +0.97 (5) | +0.30 (5) | +0.60 (5) | +0.90 (9) | +0.54 (8) |
| A29b ResNet-18 s1 | +0.68 | +0.95 | +0.97 | +0.50 | +0.60 | +0.87 | +0.22 |
| A29b ResNet-34 s0 | +0.01 | +0.60 (6) | +0.20 (9) | **−0.76** (13) | +0.09 (7) | +0.14 (17) | −0.08 (16) |
| A29b ResNet-34 s1 | +0.09 | +0.90 | +0.79 | **−0.44** | +0.14 | +0.26 | −0.18 |

讀法：agent 的第 1 條成立——ResNet-50 與 ResNet-34 的「重排」住在 stage 3（ResNet-50 另含 stage 4 的一個種子），stage 1 與 CIFAR 網路一樣保序；
但 **basic6666 兩種子的逐 stage 符號互相翻轉**（stage 2 −0.81 對 stage 3 +0.72），那個架構根本沒有一致的 stage 內讀法（第 5 條）。
公開六個 ResNet-50：§6 stage 內 +0.48／+0.55／+0.57／+0.12、依 role conv2 0.00、conv3 −0.51；/C 下 +0.57、/r_max 下 −0.08（§8 第 419–420 行）——第 2 條成立。
**stage 內、role 內的排序與分母無關**（同 stage 同 role 的 r_max/C 是常數），所以「within stage × within role」是唯一不受 M1 影響的順序統計量。

A31 K10：終態 `k*(0.95)＝1` 的層 15／29 個（s0／s1，70 列含未過門檻）、k* 值全在 1–15——第 4 條成立。K20 6 個、K50 0–1 個、K100 0 個：只有 K10 退化。

## B. 逐條

| # | 嚴重度 | 判定 | 核對與行動 |
|---|---|---|---|
| 1 | FATAL | **成立** | 表 A。全層 T–R 不得承重；3-D 改寫為「深 ImageNet 網路的晚期 stage 重排、早期 stage 保序」，basic6666 寫「種子間無一致 stage 內讀法」 |
| 2 | FATAL | **成立** | §8 419–420、§6 by role。順序主張只在 role 內、stage 內寫；conv3（1×1 擴張）的 −0.51 是 role 內的真實重排、conv2 的 0.00 是「無關係」 |
| 3 | FATAL | **部分** | 公開家族的 T–R 依參照抽樣而變（A33），不得承重；**受控訓練（A17／A28／A29b／A31／round 3）對自己初始化的同種子 T–R 是定義良好的**，是存活的核心。T–T／R–R 無參照問題 |
| 4 | FATAL | **成立** | 自算 15／29 層平手於 1 通道。P9.81、P-b9.7 撤出證據，寫「K10 格的剖面退化（中位 0.02／0.01 r_max），順序統計量不可讀」；K20／K50 不退化，照報但加 #5 的 CI。重量 K10 要重訓存 epoch checkpoint（只有 final.pt；US$3 × 2 種子，作者定） |
| 5 | MAJOR | **成立** | 表 A basic6666。zone 語言全刪；每個 T–R 給 CI（受控跑用種子，單一 checkpoint 用層的 block permutation）；P9.44 的「成立」加「效果 < 種子差」 |
| 6 | MAJOR | **成立** | CIFAR stem 與 ImageNet stem 隨資料集一起變。歸因停在「ImageNet-style training」；交叉格（CIFAR-10@224 過 ImageNet stem、ImageNet-100@32 過 CIFAR stem）US$10–15，作者定 |
| 7 | MAJOR | **成立** | 草稿 III-A(iii) 自己說 −0.3 到 +0.8。III-D 前提逐家族寫，限制在前提成立的家族 |
| 8 | MAJOR | **成立** | γ＝1 的 ResNet 系隨機對照已有（P9.58，3 種子）；role 內 T–R 兩種參照都算（免費） |
| 9 | MAJOR | **成立** | 表 A：ResNet-34 的「轉折」是 stage 3 重排、stage 1–2 保序；P9.50 照登記字面成立但措辭改成 stage 讀法；§30 重產中 |
| 10 | MAJOR | **部分** | 兩個對象（A24 ReLU 後 τ＝0.999 尺 vs A31 conv 輸出 τ＝0.95 順序）本來就分開寫；「monotone」從 κ 的陳述拿掉 |
| 11 | FATAL | **成立** | plan §9.29 自記「K＝10 的寬度已見」；只有 P-b9.2（K＝13／20）盲。B9 改「探索性觀察，K＝13／20 有斷崖的層盲測命中」，不再稱 prediction |
| 12 | MAJOR | **成立** | 同我的 A9：引理是定義；寫成觀察＋一句 remark |
| 13 | MAJOR | **成立** | P-b8.5／8.6、P-b10.1 標「post hoc，同掃描的 held-out 跑確認」，排除於成立／不成立計數 |
| 14 | MAJOR | **成立** | 同我的 A3（已用 `prereg_counts.py` 統一）；再加欄「blind／non-blind／post hoc」與一句多重比較 |
| 15 | MINOR | **成立** | §30／§31 陳舊，正在重產 |
| 16 | FATAL | **成立** | 同我的 A1／A12。III 不叫理論節；精確陳述進補充 |
| 17 | MAJOR | **成立** | B6 降為一句 remark 加四個引用（van Laarhoven、Li & Arora、Zhang–Wang–Xu、Kunin） |
| 18 | MAJOR | **部分** | Ky Fan 是 k*(τ) 的定義層級；骨架第一點縮成一句記號說明。「三種寬度」的區分保留（改措辭，見 #19） |
| 19 | MAJOR | **成立** | 同我的 A4；另外 P-d5.3（±0.5）登記為成立，−0.16 不得寫成「cost」。改「isometric kernel width」、刪準確率代價 |
| 20 | MAJOR | **成立** | 「Markov in the activation law；共變異數遞推只在線性步閉合」 |
| 21 | MAJOR | **成立** | 經驗 Fisher ≠ Fisher（Kunstner 2019）；Cui 是攤平＋截斷點。改「analogous to」 |
| 22 | MAJOR | **成立** | P-b7.2 未登記對 K 的偏相關。用既有 gradgram CSV 算 K 內 Spearman（免費、探索性標示） |
| 23 | MAJOR | **成立** | 同我的 A8；λ 寫「要有 wd；5e-4 以上不敏感」 |
| 24 | MAJOR | **成立** | B10 進補充的分解表 |
| 25 | MAJOR | **成立** | 自相矛盾。定案：對象不同（位置當樣本），塌縮定理**未被檢驗**；P 因子（4、49）當作量到的關係報 |
| 26 | MAJOR | **部分** | erank＝1 時 Ostrowski 恆等式按構造成立，實證內容是「ONI 訓練下輸入側寬度不變」；SO／SRIP 的「一半」仍是對非等距懲罰的描述，保留 |
| 27 | MINOR | 成立 | B7.1 進補充引理 |
| 28 | MINOR | 成立 | 「≥ 3K」改報絕對數；P-b7.5 的「順序全中」旁必附「數目高估 1.6–2.1×」 |
| 29 | MINOR | 成立 | `proposition-fixes` 待貼（已知） |
| 30 | MAJOR | 成立 | main.tex 貢獻 2 仍是舊句；pending 清單已有，改稿時做 |
| 31 | MAJOR | **部分** | Harun／Masarczyk 問同一組因素（`s3-prior-art` 已記）；gap 句縮成「no one measures the τ-width of the position-sampled conv-output covariance against a random draw across families」。Cohen 2020、Stephenson 2021、Galanti 2022、Recanatesi、Huh、Feng 已在 bib；**Doimo 2020、Parker 2023、Kunstner 2019、Andriushchenko 2023 未讀**，依規則先讀 |
| 32 | MINOR | 成立 | FWSVD 只當「同方向的觀察」引，不當支持 |
| 33 | MINOR | 部分 | Garg 關卡兩層差 0.2 已依規則 11 寫明；E&B 措辭已是「同一估計器兩種實作」；Yu & Wu 只寫 fc1 |
| 34 | MAJOR | 成立 | 層間自相關；p 值降為輔助、效果量＋CI 為主 |
| 35 | MAJOR | 成立 | 已知弱點 §7.4；家族層級句限於 ResNet-50 與受控訓練 |
| 36 | MAJOR | 成立 | 門檻的依據是經驗平坦化（Fig. S1），不是 Prop. 1；寫明，並對 C＝2048 層報位置數減半的穩定性（既有資料可算） |
| 37 | MINOR | 成立 | ConvNeXt dense 4 層的 p 不報 |
| 38 | MINOR | 成立 | DenseNet 0.5 的比值寫出 |
| 39 | MAJOR | 成立 | 「not with image count」錯：影像數有反向效果（P9.34 反向）；改「步數主導、影像數反向且較小」 |
| 40 | MAJOR | 成立 | 該標題候選作廢 |
| 41 | MINOR | 成立 | 反向預測寫成探索性 |
| 42 | MINOR | 成立 | T–R、λ 水準、B10 寬度都要有抖動或 CI |
| 43 | MINOR | 成立 | pending 清單（已知） |
| 44 | MINOR | 成立 | 方法段落維持一段；「讀滿 ⇔ 均勻」寫成觀察不寫成陳述 |
| 45 | MINOR | 成立 | 「simulable from second-order statistics」 |
| 46 | MINOR | 成立 | 刪 |
| 47 | MINOR | 成立 | 同 #3 |
| 48 | MINOR | 成立 | 用詞：「hypothesis written before the stored data were read」，不用 pre-registered 於這些項 |

計數：成立 40、部分 7（#3、10、18、26、31、33 與 #4 的重量部分）、不成立 0。

## C. 合併判定（我的 19 條 ＋ agent 48 條）

**過閘結果：B+ 框架（理論脊椎）不過閘。** 兩份審稿在最重的兩點一致：
1. 理論軸沒有實質定理（我的 A1、agent #16／17／18／24）；
2. 3-D「初始化的順序在深 ImageNet 網路丟失」這個全層統計量不承重（agent #1–3，我自己重算確認）。它縮成：
   **受控訓練裡，對自己的初始化，早期 stage 保序、晚期 stage 重排（ResNet-50 stage 3、ResNet-34 stage 3），ResNet-18 全保序；52 層 basic 網路種子間無一致讀法。**
   這個敘述不需要 κ 來解釋，也就沒有「上限決定順序」的機制主張。
3. A31 K10 格退化（#4），B9 非盲（#11），兩條「預測命中」都撤。

存活的東西（兩份審稿都沒打掉）：量測協定與 M1–M4；初始化剖面可由架構＋影像二階統計模擬（≥ 3 種子後）；層內律 535 層；訓練在所有家族抬高水準；
受控訓練的同種子 stage 內順序讀法（表 A）；B9 在 K＝13／20 的盲測斷崖（探索性）；摺入在少類別任務的零重訓結果（降級的應用）；164 條帶失敗的預登記表。

五軸合併（agent／我／合併）：文獻 3／4／**3.5**、題目 2.5／3.5／**3**、理論 2／3／**2.5**、資料 3.5／4／**3.5**、工具 2／3／**2.5**。
TPAMI 接受率：現框架 15–25%；改成量測論文、stage 內重分析後 30–40%（major revision 為最可能的第一輪結果）。

## D. 給作者的選項（錢與方向是作者的）

- **任何選項都要先做的（免費、Mac、約兩天）**：stage 內／role 內同種子 T–R 表（所有受控跑＋公開家族兩種參照）＋ CI；A31 K10 平手比例寫入；
  `checkpoint_analysis.txt` 重產；Table S17 加 blind／post hoc 欄；P-b7.2 的 K 內相關；pending 清單。做完後「要解釋的現象」才定型。
- **選項 1：量測論文框架**（agent 的 (c)）：無順序敘事、理論進補充；投 TPAMI 估 30–40%，TNNLS／TMLR 更高。與作者 09-16「維持 TPAMI 當代表作」、09-30「前三軸到 4.5 才投」衝突——理論軸在這份材料上到不了 4.5。
- **選項 2：繼續找 κ 的陳述**（理論軸主攻）：兩條路已關；現象縮小後要解釋的是「晚期 stage 為何重排」，比原來小。無日期、無保證。
- **選項 3：先做共同部分，再決定 1 或 2。** 我的建議。
- 付費項目（作者定）：K10 重訓存 epoch checkpoint（US$3 × 2）；stem 交叉格（US$10–15）；都只動實驗軸。
