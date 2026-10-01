# 文獻地圖（2026-09-30 起；作者的鏈：文獻 → 題目 → 理論 → 實驗 → 工具）

**目的**：以「一層的有效寬度由什麼決定」為問題，把六條相關的研究線各走到最前沿，每條線寫出它回答了什麼、留下什麼。
題目由地圖讀出來，不是宣告；理論鏈的每一環由地圖裡的定理接起來。這份文件是第 1 軸（文獻）與第 2 軸（題目）的工作檔，
逐條線完成後分別有自己的筆記（`notes/map-L<k>-*.md`），這裡只放結論與交叉引用。

**規則**（memory：literature-first-two-purposes、experiments-are-not-the-paper）：讀 PDF 原文不讀摘要；每篇兩欄產出（已有的／給我們的）；
明寫假設與對象；「未見」不等於「沒有」；不因為讀到某篇就排新實驗。

## 六條線

| 線 | 問題 | 起點（已讀或已引） | 要走到的前沿 | 狀態 |
|---|---|---|---|---|
| **L1 初始化時的譜與秩** | 隨機權重的網路，各層輸出共變異數的譜由什麼決定？（脊椎第二段「初始化時是架構」） | Saxe 2014、Poole 2016、Schoenholz 2017、Pennington–Worah 2017、Daneshmand 2020、Dong 2021、Noci 2022、Fan–Wang 2020、Louart–Liao–Couillet 2018、Everett 2026、Giorlandino–Goldt 2026 | 有限寬度、相依樣本（卷積 patch）、BN／LayerNorm、殘差尺度（γ＝0／LayerScale）對譜的影響；初始化方式作為輸入 | ◐ 第一輪 09-30 讀完 12 篇（`map-L1-initialisation-2026-09-30.md`）；第二輪待補 |
| **L2 訓練中的低秩** | 訓練把哪些層的秩壓低、由什麼決定（類別數、深度、資料、優化）？（脊椎第三段） | Papyan 2020、Rangamani 2023、Masarczyk 2023、Harun 2024、Feng 2022、Galanti 2022、Súkeník 2023／2024、Zangrando 2024、Huh 2023、Wang 2025（JMLR）、He & Su 2023 | 中間層 collapse 的理論邊界；tunnel 的因素分離；秩與類別數的定量關係；訓練早期的動力學 | ◐ 第一輪 09-30 讀完 13 篇（`map-L2-training-low-rank-2026-09-30.md`）；第二輪待補定理段 |
| **L3 對齊與特徵學習** | 訓練後的核與輸入二階統計如何對齊？這決定寬度嗎？ | Maennel 2020、Beaglehole 2023、Radhakrishnan 2024、Boix-Adserà 2025、Saxe 深線性、Ziyin、Atanasov | 對齊的時間過程；對齊對譜的後果；卷積層的 patch-AGOP | ◐ 第一輪 09-30 讀完 9 篇（`map-L3-alignment-2026-09-30.md`）；重排沒有理論，但 L1＋L2＋L3 可組裝出一條解釋鏈 |
| **L4 表徵維度的量法** | 各種「維度」量的是什麼、樣本偏差多大、彼此如何換算？ | Ansuini 2019、Roy–Vetterli 2007、Garrido 2023（RankMe）、Agrawal 2022（α-ReQ）、Elmoznino–Bonner 2024、Shawe-Taylor 2005、Blanchard 2007、Reiß–Wahl 2020、Koltchinskii–Lounici 2017、Skean 2025 | 數值秩對 τ 寬度（A34 的問題）；門檻的理論；跨量法的換算 | ◐ 第一輪 09-30（`map-L4-measures-2026-09-30.md`；Table S1 已覆蓋量法本身，本輪補 Kong–Valiant 少樣本估譜、冪律譜上兩把尺的閉式） |
| **L5 寬度的選擇與壓縮** | 寬度怎麼被選、學、剪；學出來的寬度長什麼樣？ | Garg 2019／2024、Yu & Wu 2023、Chee 2022、CHEX、DMCP、BCNet、Network Adjustment、Width Transfer、Yuan 2023、NORTH\*、Liu 2019 | 逐層寬度的規律（後段寬）、成本項對訊號項、無標籤判準 | ◐ 第一輪 09-30（`map-L5-width-selection-2026-09-30.md`）：已知很多，留下「無標籤尺對有標籤判準的差」、「成本項對訊號項」、「寬度 ≠ 秩」 |
| **L6 殘差與 skip** | 殘差流對譜與秩做了什麼？ | Dong 2021、Li & Papyan 2023、Zhang & Jiang 2024、Everett 2026、Haggi-Mani & Rish 2026、Daneshmand 2020 | 加法處的譜（我們的 §10.17）；skip 對訓練後寬度的作用；BN 與 skip 的分工 | ◐ 第一輪 09-30（`map-L6-residual-2026-09-30.md`）：初始化的殘差理論（Yang–Schoenholz、Daneshmand）與訓練後的 Residual Alignment 已知；加法處的譜、skip 對訓練後寬度的作用沒有 |

## 交叉的問題（地圖畫完才能答）

1. 六條線都留下的問題是哪一個？——那是題目。
2. 哪些定理能接成一條鏈（初始化的譜 → ReLU 步 → 訓練改變什麼 → 殘差的下界）？每一環的假設在卷積網路上成立到哪裡？
3. 我們的每個實驗驗證的是鏈上哪一環？放不進去的實驗要問為什麼還在。

### 第一輪後的暫定回答（09-30 晚；第二輪讀完再改）

**1. 六條線都留下的問題：** 「一個訓練好的網路，**逐層**的表徵寬度（譜的形狀）由什麼決定、怎麼算」。
- L1 給初始化（架構＋初始化方式決定，理論只有一個相關係數或一個下界，沒有逐層譜的形狀）；
- L2 給訓練後低秩的三個旋鈕（深度、wd、類別數）與時間（TPT），但只說「塌縮層」，不說哪些層、多少；
- L3 給對齊與譜的三種變化，但沒有逐層順序；
- L4 給量法，但沒有量法之間的換算；
- L5 給寬度的選法，但把寬度當成本問題，不問它由什麼決定；
- L6 給殘差保幾何的理論，但沒有加法處的譜。
每一條線都在自己的量上停在「全網一個數」或「最後一層」，**逐層剖面這個對象是空的**。這是題目該站的地方。

**2. 能接成鏈的定理（每環附假設；卷積上成立到哪裡是我們的量測）：**
初始化譜＝尖峰＋本體（Yang 2019；BN、無限寬）→ PR ≥ Ω(√C)（Daneshmand；線性、BN）→ 層內律 S6（Silverstein；給定 Σ_patch）→ ReLU 步（Fan–Wang／LLC；獨立樣本，**卷積 patch 不滿足**）
→ 早期對齊去尖峰（Atanasov；小初始化）→ 後期 DNC 壓塌縮層到 ≈ K（Súkeník／Zangrando；UFM、wd）＋深度偏好（Jacot；L→∞）→ 殘差加法處的下界（Weyl／Ky Fan；任何共變異數）。
**重排＝這條鏈的組裝後果**（L3 §3），不需要新定理，但每一環的「在卷積網路上成立到哪裡」都要用我們的資料驗證——那就是實驗的角色。

**3. 實驗對到鏈上的位置：** 44 權重與家族表→初始化譜與 R–R（環 1–2）；層內律 535 層（環 3）；copula 診斷（環 4）；ρ_align 時間過程（環 5）；A24／A25／A31（環 6 類別數與 TPT）；A17／A28／A29b（環 6 深度）；A33（環 1 初始化方式）；§10.17（環 7）。
**放不進去的**：控制器（P2 系列）、零重訓摺入的準確率數字、A20 的表徵診斷——這些是 L5 的應用，論文各留一段或移到補充。

**4. 前三軸現在的估計**：文獻 3（六線第一輪完成，第二輪與 bib 未做）；題目 3.5（有了「逐層剖面由什麼決定、怎麼算」這句，但要用第二輪確認沒人做）；
理論 2.5（鏈的每一環都是別人的定理；「組裝」還沒寫成嚴謹的陳述）。

## 題目的空格：2025–2026 的搜尋（09-30 晚，三次查詢）

- **Everett 2026**（2607.14018，「Transforming Rank」）：已在 L1／L6；transformer 前饋 block 各元件在**初始化**時保留多少秩；對象是梯度與表示的有效秩、單一深度剖面。
- **Makwana 2025**（2507.07675）：有限寬全連接 ReLU 網路對固定 m 個輸入的隱層激活矩陣**代數秩**期望值的閉式（幾何衰減、「復活」深度）；單作者、8 頁、
  文中殘留 LLM 引用標記，可信度待查。對象是代數秩，不是譜的形狀；與我們的量不同，但主題相近，**引用前要核**。
- **Pérez-Corral et al. 2026**（2603.21991，λ-GELU）：用 Spearman 比較不同初始化下的逐層「gating hardness」剖面，結論「訓練誘導的層排序對初始化穩健」——
  **同一個統計量的想法**（跨初始化的逐層順序相關）用在另一個量上；要引，並說明我們的量是譜寬度、比較的是訓練前後而非初始化之間。
- 「hunchback／hourglass」ID 剖面在 LLM 上的近作（Yusupov 2025、SISSA 論文 2025）：都是實證，明寫「existing studies are exclusively empirical—they describe
  geometric properties but do not explain why these properties arise」——**這句話就是我們題目的空格，別人替我們寫了**。
- **結論（暫定）**：「逐層譜寬度剖面由什麼決定」沒有理論工作，實證只有單一剖面或單一統計量；跨訓練前後的逐層順序相關、多配方一致性、初始化方式的角色
  仍未見。三次查詢不是窮盡；Masarczyk 的 39 篇被引已掃（09-30 晚：新增 2505.15239 DNC 推到 ResNet／transformer、2607.27854 訓練中簡化網路），Ansuini、Feng、Daneshmand 的被引待掃（API 限流）。

## 進度

| 日期 | 做了什麼 |
|---|---|
| 09-30 | 建檔；L1 第一輪：Poole、Schoenholz、Xiao、Daneshmand、Yang、Hanin–Rolnick、Novak、Martens、Fan–Wang、LLC、Dong、Noci／Giorlandino–Goldt（摘要）。結論：隨機深網失去維度、BN 擋住（PR ≥ Ω(√d)）、位置相關的傳遞、初始化譜＝尖峰＋本體都有理論；洞是有限寬度＋卷積＋BN＋ReLU 同時在場時譜的形狀 |
| 09-30 | L2 第一輪：Jacot 2023、Timor 2023、Galanti、Súkeník／Zangrando、Wang JMLR、He & Su、Papyan、Rangamani、Parker、Masarczyk／Harun、Feng。結論：低秩的三個旋鈕（深度、wd、類別數）有理論；哪些層塌、剖面形狀、對初始化順序的關係、資料因素沒有理論。我們每個 S3 實驗可以對到一條陳述（A24／A31→DNC、A25→TPT、A29b→Jacot），重排（T–R<0）沒有 |
| 09-30 | L3 第一輪：Saxe、Maennel、Beaglehole、FACT、Atanasov（silent alignment）、Ba、Wang 2023（本體不變／尖峰／重尾）、Martin–Mahoney、Kunin。結論：對齊已知、譜的三種變化已知（兩層）；重排沒有理論；**候選的組裝解釋**：初始化尖峰＋本體（L1）→ 早期對齊去尖峰、水準升（L3）→ 後期 DNC 壓後段（L2）→ 塌縮段長（深、少類）時順序反轉。三條可用既有資料驗證的陳述寫在 L3 §3 |
| 09-30 | L4 第一輪：Table S1 的 20 篇＋Kong–Valiant 2017、Ghosh 2022、RankMe、Recanatesi。結論：量法與偏差已知；門檻可由少樣本估譜取代（Kong–Valiant）；純冪律譜上數值秩讀頭部、τ 寬度讀滿——A34 的 0.2 對 0.5 說明真實譜非純冪律。留下：量法換算、位置相依樣本的估計理論 |
| 09-30 | L5、L6 第一輪由既有筆記整理（Liu 2019、Veit、Balduzzi、Yang–Schoenholz 新讀）。六條線第一輪完成，見 §「交叉的問題」的暫定回答 |
| 10-01 | 第二輪第一批（`b7-prior-art-2026-10-01.md`）：四次搜尋、11 篇下載、6 篇新讀。**B7.1 記憶窗已知**（Wang & Aitchison 2025；Everett & Qiu 2026），我們的只剩 B7.2′＋量測；Fan 2025 反向預測 P-b7.4d；L2 補 Garrod 2026（深度給 NC）、Beaglehole 2024（AGOP ⇒ DNC）。scan0930 的五篇「未讀」其實 09-30 已讀，Dees 2022 丟。剩：L4 待讀清單、L5／L6 定理段、各線 2025–26 第二批 |
| 10-01 | 第二輪第二批（`literature-round2-2026-10-01.md`）：L2 補 6 篇 2024–26（Zhao ICLR 2026 優化器、Jacot 2024、PFC、Rupa、Garrod、Beaglehole）；L4 待讀三篇讀完；題目的兩個最近鄰出現（NSC：規格→權重譜標量；Spectral Lens：激活＋梯度譜診斷），都不是逐層激活剖面。第 1 軸仍 3.5（劃界句與 bib 未進稿） |
| 10-01 | 第三批：L3 加 Beaglehole–Mitliagkas–Agarwala TMLR 2024（NFA ＝ 對輸出端梯度特徵的對齊——A32 的理論出處）、Nyström 2026（交換子框架，旁引）；L6 定理段讀完（Yang–Schoenholz 多項式速率、Balduzzi 定理 3、Li–Papyan 定理 3.1／3.3）；L5 的 2025–26 寬度學習查詢只回到 AWN（已知）。**第二輪收尾**：剩 L2 定理段的最後一級（Súkeník 2025 定理 4.3）與投稿前的再查。第 1 軸 3.5 不變；進稿子的劃界句才是下一級 |
| 10-01 | Súkeník 2025 定理 4.3／推論 4.5 的假設讀到第一級（map-L2 末節）。**第二輪關閉**；剩投稿前一週的重查與補充的「搜尋紀錄」一節（radar 補強計劃 1(d)）。獨立的敵意審稿 agent 已派（結果 `notes/hostile-review-agent-2026-10-01.md`） |

