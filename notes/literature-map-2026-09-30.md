# 文獻地圖（2026-09-30 起；作者的鏈：文獻 → 題目 → 理論 → 實驗 → 工具）

**目的**：以「一層的有效寬度由什麼決定」為問題，把六條相關的研究線各走到最前沿，每條線寫出它回答了什麼、留下什麼。
題目由地圖讀出來，不是宣告；理論鏈的每一環由地圖裡的定理接起來。這份文件是第 1 軸（文獻）與第 2 軸（題目）的工作檔，
逐條線完成後分別有自己的筆記（`notes/map-L<k>-*.md`），這裡只放結論與交叉引用。

**規則**（memory：literature-first-two-purposes、experiments-are-not-the-paper）：讀 PDF 原文不讀摘要；每篇兩欄產出（已有的／給我們的）；
明寫假設與對象；「未見」不等於「沒有」；不因為讀到某篇就排新實驗。

## 六條線

| 線 | 問題 | 起點（已讀或已引） | 要走到的前沿 | 狀態 |
|---|---|---|---|---|
| **L1 初始化時的譜與秩** | 隨機權重的網路，各層輸出共變異數的譜由什麼決定？（脊椎第二段「初始化時是架構」） | Saxe 2014、Poole 2016、Schoenholz 2017、Pennington–Worah 2017、Daneshmand 2020、Dong 2021、Noci 2022、Fan–Wang 2020、Louart–Liao–Couillet 2018、Everett 2026、Giorlandino–Goldt 2026 | 有限寬度、相依樣本（卷積 patch）、BN／LayerNorm、殘差尺度（γ＝0／LayerScale）對譜的影響；初始化方式作為輸入 | ☐ 進行中（09-30 起） |
| **L2 訓練中的低秩** | 訓練把哪些層的秩壓低、由什麼決定（類別數、深度、資料、優化）？（脊椎第三段） | Papyan 2020、Rangamani 2023、Masarczyk 2023、Harun 2024、Feng 2022、Galanti 2022、Súkeník 2023／2024、Zangrando 2024、Huh 2023、Wang 2025（JMLR）、He & Su 2023 | 中間層 collapse 的理論邊界；tunnel 的因素分離；秩與類別數的定量關係；訓練早期的動力學 | ☐（`s3-prior-art-2026-09-30.md` 是起點） |
| **L3 對齊與特徵學習** | 訓練後的核與輸入二階統計如何對齊？這決定寬度嗎？ | Maennel 2020、Beaglehole 2023、Radhakrishnan 2024、Boix-Adserà 2025、Saxe 深線性、Ziyin、Atanasov | 對齊的時間過程；對齊對譜的後果；卷積層的 patch-AGOP | ☐（`alignment-prior-art-2026-09-29.md` 是起點） |
| **L4 表徵維度的量法** | 各種「維度」量的是什麼、樣本偏差多大、彼此如何換算？ | Ansuini 2019、Roy–Vetterli 2007、Garrido 2023（RankMe）、Agrawal 2022（α-ReQ）、Elmoznino–Bonner 2024、Shawe-Taylor 2005、Blanchard 2007、Reiß–Wahl 2020、Koltchinskii–Lounici 2017、Skean 2025 | 數值秩對 τ 寬度（A34 的問題）；門檻的理論；跨量法的換算 | ☐（`th2-th5-prior-art` 與 A34 是起點） |
| **L5 寬度的選擇與壓縮** | 寬度怎麼被選、學、剪；學出來的寬度長什麼樣？ | Garg 2019／2024、Yu & Wu 2023、Chee 2022、CHEX、DMCP、BCNet、Network Adjustment、Width Transfer、Yuan 2023、NORTH\*、Liu 2019 | 逐層寬度的規律（後段寬）、成本項對訊號項、無標籤判準 | ☐（`related-work-boundary-2026-09-22.md`、`p2m-prior-art-2026-09-30.md` 是起點） |
| **L6 殘差與 skip** | 殘差流對譜與秩做了什麼？ | Dong 2021、Li & Papyan 2023、Zhang & Jiang 2024、Everett 2026、Haggi-Mani & Rish 2026、Daneshmand 2020 | 加法處的譜（我們的 §10.17）；skip 對訓練後寬度的作用；BN 與 skip 的分工 | ☐（`th9-prior-art-2026-09-29.md` 是起點） |

## 交叉的問題（地圖畫完才能答）

1. 六條線都留下的問題是哪一個？——那是題目。
2. 哪些定理能接成一條鏈（初始化的譜 → ReLU 步 → 訓練改變什麼 → 殘差的下界）？每一環的假設在卷積網路上成立到哪裡？
3. 我們的每個實驗驗證的是鏈上哪一環？放不進去的實驗要問為什麼還在。

## 進度

| 日期 | 做了什麼 |
|---|---|
| 09-30 | 建檔；L1 開始 |
