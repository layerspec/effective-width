# 題目陳述（一頁草稿；2026-09-30 晚，六線第一輪＋部分第二輪後；**給作者定，不是定案**）

## 問題

**一個深度網路每一層的表徵寬度——輸出共變異數的譜要多少方向才裝得下——由什麼決定？能不能在訓練前算、在訓練中預測它怎麼變？**

「寬度」在這裡是譜的量（達 τ 變異的方向數除以可達秩），不是通道數、不是流形內在維度、不是數值秩；四者的關係是問題的一部分（L4）。

## 為什麼現在沒有人答

六條研究線各自碰到這個對象的一面，但都停在「全網一個數」或「最後一層」：

| 線 | 有的 | 停在 |
|---|---|---|
| 初始化的訊號傳播（L1） | 相關係數的不動點、BN 的 PR 下界、位置相關的傳遞 | 一個相關係數或一個下界；沒有逐層譜的形狀 |
| 訓練中的低秩（L2） | 深度、weight decay、類別數三個旋鈕；TPT | 「塌縮層」的秩 ≤ K；不說哪些層、每層多少 |
| 對齊與特徵學習（L3） | 對齊的存在與時間；譜的三種變化 | 全網共享的模或單層；沒有逐層順序 |
| 維度的量法（L4） | 各量法與偏差 | 沒有量法之間的換算、沒有相依樣本的理論 |
| 寬度的選擇（L5） | 學出來的寬度贏均勻、形狀是後段寬 | 把寬度當成本問題，不問它由什麼決定 |
| 殘差（L6） | 初始化保幾何、訓練後支路低秩 | 加法處的譜沒人量 |

近作自己承認：「existing studies are exclusively empirical—they describe geometric properties but do not explain why these properties arise」（Yusupov et al. 2025）。

## 答案的形狀（理論鏈，每環是已知定理，假設明寫）

1. **初始化**：譜由（架構、初始化方式、影像的二階統計）決定——位置共變異數的遞推（Xiao）給無限通道極限；層內律（Silverstein）給有限寬度的精確版；
   BN 給下界 PR ≥ Ω(√C)（Daneshmand）；初始化尺度（γ、LayerScale、權重方差）是輸入之一（Yang–Schoenholz、Kunin）。
2. **訓練早期**：對齊在 loss 未降時完成（Atanasov），尖峰被抹平、每層水準上升。
3. **訓練後期**：weight decay＋類內變異小 → 塌縮層的權重接近秩 K（Zangrando）；深且多類時更低（Súkeník）；深度本身偏向低秩（Jacot）；發生在 TPT（Papyan）。
4. **淨效果**（09-30 深夜依既有資料更正）：類別少 → 後段被塌縮壓在低處、順序保留；類別多且深 → 後段不受壓、從最窄升到最寬、順序反轉。**重排是組裝後果；
   真正開放的是「後段的上限由什麼定」——DNC 說 ≈ K，量到的是 K 的 2–50 倍。**（09-30 下午的候選答案 B7：權重是梯度的記憶窗、窗在終端相之前；κ ＝ 窗內仍有 EL2N 的樣本 × 遮罩多樣性——等 gradgram 判。）
5. **殘差**：加法處的絕對譜不減（Weyl），τ 寬度在 ReLU 前變窄、ReLU 後回升；支路低秩（Li & Papyan）。

## 實驵的角色：驢證每一環在真實卷積網路與 transformer 上成立到哪裡

44 個公開權重與家族表（環 1：R–R、初始化方式）；層內律 535 層（環 1）；copula 診斷（環 1 的 ReLU 步在相關 patch 上失效多少）；
ρ_align 的時間過程（環 2）；A24／A25／A31（環 3：類別數、TPT）；A17／A28／A29b（環 3：深度）；A33（環 1：初始化方式）；§10.17（環 5）；
A30（同一條鏈在 transformer 上）。**放不進鏈的**：控制器、摺入的準確率、A20——各一段或進補充。

## 10-01 更新（第二輪第一、二批之後；判定：A28、A27、A35、B7、B2）

- **最近的鄰居（第二輪找到，題目陳述要寫進「為什麼沒有人答」）**：Zhu–Zhao–Lu 2026 NSC——從架構規格用 MP 律閉式算**權重**譜的容量**標量**（transformer 架構搜尋）；
  Liu–Paquette–Sous 2026 Spectral Lens——**激活共變異數譜與逐樣本梯度譜並列**當 LLM 訓練診斷。兩者各佔我們問題的一半（規格→譜；激活譜對梯度譜），
  都不是「逐層激活剖面由什麼決定」；劃界句：「NSC scores the weights from the spec; we compute the activation profile given the image statistics」、
  「Spectral Lens reads the two spectra side by side; we relate them」。
- **第 3 環改寫**：塌縮的定量界在 conv–BN 層沒有駐點可咬（B6，A35 驢證：|cos| ≤ 0.04、梯度是 λW 的 1.3–13 倍）；塌的是池化後的表徵（wcv 0.07），
  不是權重（尾能量 0.4–0.6）也不是位置取樣的 τ 寬度；κ 跟輸出端梯度 Gram 的寬度走（+0.86）、Γ 隨步數縮、頭部順序可前向算。
  **記憶窗本身已知**（Wang & Aitchison 2025），我們的只剩 B7.2′ 的界與量測——今晚的 P-b7.1′／4 決定它是「界」還是「順序」。
- **第 4 環改寫**：ImageNet 內一百類、64 像素、每類 130 張都不恢復順序，**步數把順序推向重排**（A28）；歸因停在「資料在這個深度」；
  plain／殘差的二分法撤回（A27）；最後一格 A31（十類 × 224²）在跑。
- **第 5 環**：B2 關閉（下界只到 0.24），只留 Weyl。
- 問題句、六線表、三個標題候選**不變**。

## 題目候選（作者定）

- *What Sets the Width of a Layer: The Spectrum of Deep Representations from Initialisation to Collapse*
- *Layer Width Is Not a Hyperparameter: How Architecture and Task Fix the Effective Width of Every Layer*（較大，但「不是超參數」的方法部分已降級，慎用）
- *The Layer-Wise Width Profile of Deep Networks: Initialisation Sets the Order, the Task Sets the Level*（最貼資料）

## 與 B+ 的差別

脊椎相同（初始化是架構、訓練中是任務），但**理論先行**：先寫五環的鏈與假設，實驵按環排列；重排從「發現」改為「組裝後果」；transformer 進正文；
控制器一段。第 2 軸能否到 4.5，取決於第二輪確認空格為真、以及鏈能否寫得嚴謹（第 3 軸）。

## 還沒做、決定題目前要做的（10-01 更新：第 1 項的第二輪已做兩批，剩 L5／L6 定理段與各線第三批；第 3 項仍未跑）

1. 第二輪：鏈上每一環的定理讀到第一級（目前 Yang、Daneshmand、Zangrando、Súkeník、Jacot、Ba、Wang、Atanasov 到第二級；Xiao、Poole 到第二級）。
2. ~~Semantic Scholar 引用圖掃一次（Ansuini、Masarczyk、Feng、Daneshmand 的被引），確認空格。~~ **09-30 下午已掃**（Ansuini 457、Feng 64、Recanatesi 86、
   Daneshmand、Chen、Yu&Wu、Joudaki、Garg、Everett、Bayat 的被引；`results/lit_scan_citations_0930.json`）：**空格仍在**。最接近的：Chen–Bu–Yan NeurIPS 2024
   （逐層「寬度夠不夠」的權重範數判準，實證、要到收斂才判）、Bayat 2026 TLM 與 Mudarisov 2026（LLM 逐層 MLP 寬度配置，實證／幾何量、無初始化預測）、
   Garg 2024（utilized rank）、Everett 2026（transformer 初始化 Jacobian 秩）——沒有一篇問「逐層寬度由什麼決定、能否訓練前算」。
   理論端新增：Joudaki 2023×2＋Joudaki–Hofmann 2024（環 1′：BN 等距、忘掉輸入、殘差減慢）、Garrod–Keating 2025（環 3：最優秩 ≤ 2、DNC 隨寬度更常見；
   **沒有理論預測 κ ≫ K**）。題目陳述不變；「為什麼現在沒有人答」表的 L2、L5、L6 列各加一句（見各地圖）。
3. 三條組裝解釋的驢證（`map-L3` §3；既有資料）——若不成立，第 4 環要改寫。
