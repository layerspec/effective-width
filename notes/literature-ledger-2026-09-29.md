# 文獻總帳（2026-09-29 起；作者規則：文獻查詢有兩個用途——不重做、取得改進靈感）

每一列兩欄產出：**已經有人做的**（引用，不重做）與**給我們的靈感**（可以改的設計、對照、統計量、說法）。
細節在各查核筆記；這裡只放結論與行動。狀態：☐ 未做、☑ 已做、◐ 已排。

| 文獻 | 已經有人做的 | 給我們的靈感 | 行動 |
|---|---|---|---|
| Shawe-Taylor 2005；Blanchard 2007；Reiß–Wahl 2020 | 樣本特徵值前 k 和的上偏、√(k/n) 界、置中情形、冪律譜下更緊的界 | **樣本門檻不必是固定的 n/C ≥ 50**：用 Blanchard 的 relative bound 從該層自己的譜算出「這一層要多少樣本才能把 k* 估到 ±1 通道」，門檻變成資料相依、有定理背書 | ☐ 在 44 個權重的譜上算；若與 50 差不多就寫一句，若差很多就改門檻 |
| 同上＋Koltchinskii–Lounici 2017 | 高斯樣本共變異數的集中，由 effective rank r(Σ) 決定 | **「寬度就是樣本數」**：初始化的一層＝對輸入做 C 個樣本的 PCA。S2 可以用一句話講清楚，並預測初始寬度的波動 ∝ 1/√(C·r(Σ)) | ☐ 用 `results/freeprob/` 的譜驗證波動的尺度 |
| Maennel 2020 | 第一層權重與資料主成分對齊（命題＋量測）；用輸入特徵向量建構深層初始化，重現隨機標籤預訓練的加速 | (i) 增益曲線「先升後降」——我們只報一個 Spearman 係數，看不到頂端；(ii) **資料相依初始化加速訓練但不提高最終準確率**，正好解釋我們先驗起點臂「無增益」的負結果，可以引用來定位 | ◐ A32 存逐方向的值（L3）；☐ 先驗臂的段落引 Maennel |
| Beaglehole 2023；Radhakrishnan 2024 | WᵀW ∝ patch-AGOP，每一層、ImageNet 公開權重 | **以梯度加權的寬度**：現在的 k* 由變異決定、不看任務（9 月 2 日文獻地圖就寫過「深層判準必須是監督式的」）。用 Γ 加權的輸出共變異數算 k*，得到「對輸出有影響的寬度」，可能正是 VGG／CIFAR 以外「尺讀滿但仍可砍」的解釋 | ◐ A32 已存 Γ；☐ 跑完後探索性地算梯度加權的 k*（標示探索性） |
| **Wang & Aitchison ICML 2025；Everett & Qiu 2026** | AdamW／decoupled wd 的權重＝最近更新的 EMA，時間尺度 1/(ηλ)；momentum 與 wd 都是「記憶」 | **B7.1 的記憶窗已知**；我們的是精確脈衝響應（momentum／Nesterov／時程）＋列空間 ⇒ Σ_out 的 τ 寬度（B7.2′）。論文措辭見 `b7-prior-art-2026-10-01.md` §1 | ☐ 改稿時引；bib 核 Crossref |
| Fan et al. 2025（2510.15262） | 縫放不變架構穩態下權重奇異值譜 ∝ √(η/λ)、形狀不變 | **P-b7.4d 的反向預測**（λ 不改 τ 寬度）；判定時引 | ☐ P-b7.4 判定段引 |
| Apte 2026（2607.23777） | 梯度 ⟂ 權重（縫放不變）、momentum buffer 對權重負對齊、常數 wd 無駐點（Prop. 3.1） | B6 第四個引用；proofs 筆記加「buffer 負對齊」一句 | ☐ |
| Garrod–Keating–Thrampoulidis 2026（2605.23087） | 深 UFM 不靠 L2 就有 NC；softmax codes 秩低於 K−1 | 第 3 環劃界：B6 排除的是 Zangrando 的量化界，不是塌縮本身 | ☐ 第 3 環一句 |
| Beaglehole–Súkeník–Mondelli–Belkin NeurIPS 2024（2402.13728） | AGOP 投影是 DNC 的資料相依機制（Deep RFM 定理 5.3） | 接 A32（AGOP）與第 3 環；B7.3 是它在位置解析梯度上的補充 | ☐ `sec:decompose`、第 3 環引 |
| **Liu–Paquette–Sous 2026（Spectral Lens）** | LLM 的激活共變異數譜＋逐樣本梯度譜當診斷 | B7 的最近鄰：他們並列、我們關聯（記憶窗、列空間、τ 寬度） | ☐ B7 段劃界句 |
| **Zhao–Cheng–Masarczyk–Lucchi ICLR 2026** | AdamW 不到 NC、SGD／Adam 會；momentum 加速 | 塌縮陳述寫明優化器；跨配方 T–T 0.86 對優化器穩健；B6 限 coupled wd | ☐ 第 3 環、配方段 |
| Jacot–Súkeník–Wang–Mondelli 2024 | 末端 ≥ 2 線性層的寬網路＋wd 證 NC | 定理對象是末端線性層，解釋 A35「池化特徵塌、conv 輸出不塌」 | ☐ 第 3 環一句 |
| Wang–Gai–Zhang 2024（PFC）；Rupa 2026 | ResNet 中間層漸進塌縮；NC 起點＝特徵範數門檻 | 中間層 collapse 的實證鄰居；「塌縮前」的定義 | ☐ 第 3 環引 |
| **Zhu–Zhao–Lu 2026（NSC）** | 從架構規格用 MP 律閉式算權重譜的容量標量 | S2 的最近鄰：權重 vs 激活、標量 vs 剖面；(i) 段劃界、題目陳述加 | ☐ |
| Gao–Ganguli 2017；Del Giudice 2021；Stringer 2019 | PR 理論與 NTC 上界；ED 教程；冪律譜與平滑性 | Discussion 一句（維度由任務上界）；Table S1 出處；冪律是預期的 | ☐ |
| Fernando–Guitchounts 2026 | LLM 殘差流 Jacobian 譜幾何 | L6 鄰居（Jacobian 不是共變異數） | ☐ |
| **Beaglehole–Mitliagkas–Agarwala TMLR 2024** | NFA ⇔ W 的左奇異結構對輸出端梯度特徵（PTK）的對齊；SGD 驅動 | A32 結果的理論出處（增益跟梯度能量走）；`sec:decompose` 引 | ☐ pending §C 加一句 |
| Boix-Adserà 2025（FACT） | 含 weight decay 的一階最適條件 | BN 網路上該條件不成立（梯度 ⟂ 權重）——劃界句；無正規化的網路（VGG 無 BN 版）可以當檢驗對象 | ☐ 改稿時寫劃界句 |
| Li & Papyan 2023（Residual Alignment） | 訓練後 ResNet 支路 Jacobian 的秩 ≤ 類別數；去掉 skip 就消失 | **類別數 → 支路的秩 → 流最多失去多少方向**（配 Th9(a)）。A26 的十類 ResNet-18 權重在 pod 上，量它的支路寬度即可直接檢驗 | ☐ 殘差流量測加 A26 權重（pod 或把 final.pt 拉回 Mac） |
| Dong 2021；Noci 2022；Daneshmand 2020 | skip 緩解秩塌縮；深度趨於無窮仍塌縮；緊下界是公開問題 | 我們的下界依賴量得到的 ρ，可以回應「公開問題」但要寫明逐 block、隨深度變鬆 | ◐ 殘差流量測已排 |
| Everett 2026 | Transformer 前饋 block 初始化時 skip 如何保秩；支路／捷徑比值是旋鈕 | γ＝0／γ＝1 的對照就是這個旋鈕的兩端；可以把「初始剖面對支路尺度的依賴」寫成 S2 的一部分 | ◐ T6 已排 |
| Zhang & Jiang 2024 | 衰減恆等捷徑 → 特徵秩下降、探針變好 | **對殘差的介入實驗**：像 ONI 之於正交，衰減捷徑之於 Th9——預測寬度怎麼變，再量。候選實驗 | ☐ 視 A27／殘差量測結果決定 |
| Haggi-Mani & Rish 2026 | 逐位置與池化兩種量法結果不同；訓練後 MLP 殘差流降秩 | 我們的池化結果有獨立的旁證，引用；「殘差保秩」不能寫全稱 | ☐ 改稿時引 |
| **Harun et al. 2024**（已在 refs.bib） | **64 個 DNN、512 個實驗：解析度（32／64／128／224）、類別數、每類樣本數、增強、深度、stem 對 tunnel effect 的影響**；類別數與解析度都減少深層壓縮；成對 Wilcoxon＋SHAP 排序 | **A28 的三個因素他們都動過**，只是量的是線性探針的 tunnel（數值秩在附錄）。(i) 他們拆類別數與影像數的做法：**固定總影像數、改類別數**（10×1000、50×200、100×100），比我們的 K1000-N130 更乾淨；(ii) 他們發現低解析度加長 tunnel——**我的 P9.42（解析度不恢復順序）依文獻可能不成立**；(iii) 增強是第二大因素，我們的 A28 沒有動它 | ☑ plan §9.17 補記（今天，A28 開跑前）；☐ A31 劑量反應採用「固定總影像數」設計 |
| Masarczyk 2023（已在 refs.bib） | tunnel 隨類別數變；**去掉 skip 的 ResNet 出現 tunnel** | A27（plain 對殘差）有先例：他們用線性探針，我們用 τ＝0.999 的尺；要引並劃界 | ☐ 改稿時引；A27 判定時對照 |
| Trockman, Willmott & Kolter 2023 | 卷積濾波器共變異數的結構，用來做初始化 | 與 Maennel 的建構同一路；先驗起點臂的相關工作 | ☐ 讀原文後決定是否引 |
| **Yu & Wu 2023**（AAAI；09-30 讀，未引） | DeiT-B 每個 block 的 QKV／FC1 輸出共變異數、90% 能量的維度比例（隨深度 0.06→0.6）；AFM＝PCA 截尾摺入、2,000 張無標籤影像、不微調，砍 20% 參數 −0.09 | **transformer 的驗證關卡**：同權重、同統計量，A30 已量；分母依推論是 min(m, n)＝r_max | ☑ 登記 P9.67（plan §9.19 補記，A30 結果未讀）；☐ 改稿：(ii) 的 transformer 段寫成「運算子是 Yu & Wu 的」 |
| **Garg, Koguchi, Verma & Ulbricht 2024**（arXiv 2407.04797；09-30 讀，未引） | 逐層 utilized rank＝P_S Wᵀ P_T 的秩 ÷ min(m, d)；ImageNet ViT-B/16 MLU 35%、ResNet-50 60%；VGG-16 CIFAR-10 對 CIFAR-100 參數比 0.18 對 0.61；拆層後微調 | (i) **r_max 型分母已有人用**，承重主張第一列要改寫；(ii) 「ImageNet 沒有閒置寬度」只在 τ＝0.999、不重訓下成立；(iii) 輸入／輸出兩端的寬度分開報，我們的分解已有原料 | ☐ 改稿：r_max、(ii)、(iv) 三處引用並劃界；☐ 探索性：用 A32 的 Γ 算任務相依寬度，與 MLU 對照 |
| Tian, Xu & Li 2026；Park 2023；Skean 2025 | 最後一層：每張影像低秩、資料集不低秩；訓練目標決定譜（MAE 最後一層 235/768、DINO 624/768） | A30 的 DINO／MAE 探索性權重對 Tian 表 1 核一次；劃界句「我們量的是跨影像的通道共變異數，不是序列內 token 的秩」 | ☐ A30 判定時核；☐ 改稿時寫劃界句 |
| **Masarczyk 2023 × Harun 2024，逐條對照我們的實驗**（09-30；`notes/s3-prior-art-2026-09-30.md`） | 類別數、影像數、寬度、解析度、深度、去掉 skip 對壓縮的影響，方向全部已報（探針與數值秩，CIFAR 與 ImageNet-100）；深層的秩在 75 步內塌縮；公開 ImageNet-1k 權重只有 ResNet-50 有 tunnel | (i) **兩把尺並列**：他們的數值秩對我們的 k*(0.999)，解決「75 步」對「要優化夠久」的衝突；(ii) 重排發生在深度的哪一段；(iii) 文獻的空格：少類別 × 224²；(iv) 文獻留下的問題：ResNet 不論有無 skip 秩都高於 VGG | ☐ (i) 先寫預登記再量（Mac，約半小時）；☐ (ii) 現有資料、探索性；☐ 改稿：(iv) 的開頭先引這兩篇 |
| **ImageNet ResNet-18 在 1.0–1.2 GFLOPs 的寬度方法**（09-30，P2m 開跑前；`notes/p2m-prior-art-2026-09-30.md`）：CHEX、DMCP、TAS、BCNet、Network Adjustment | 同一點上 CHEX 一次訓練 69.6、DMCP 69.2；均勻縮窄 68.8（1.2G）／67.1（0.9G），學出來的寬度贏均勻 2.5–2.8 點；形狀是把 FLOPs 給後段 | **成本項與訊號項要分開**：P2m 的兩臂設計贏了也無法歸因；CIFAR 的 FLOPs 守恆結果也要問同一題 | ☐ 作者定 P2m：取消／照跑／加歸因臂；☐ P2g 現有資料算「只看成本」的寬度對控制器寬度 |
| **Th8 ReLU 步的譜律**（09-30；`notes/th8-prior-art-2026-09-30.md`）：Louart–Liao–Couillet 2018、Fan–Wang 2020、Adamczak 2025、Schröder 2023、Cranston 2026 | Fan–Wang：多層 CK 譜＝MP 映射∘（線性縫合＋白噪聲），**假設樣本近似正交**；LLC：X 任意的確定性等價（解不動點） | 09-27 三個閉式失敗的原因就是 3.2(c)：相鄰位置的 patch 相關。可做：用 LLC 定理 1 在相關 patch 上驗一次適用範圍；把量到的 ReLU 步當黑盒接在 S6 後面得到可算的整條剖面 | ☑ 查完；☐ 改稿：脊椎第二段引兩篇並寫明假設在卷積 patch 上不成立；☐（選配）LLC 不動點驗證 |
| **Nguyen, Raghu & Kornblith 2021**（ICLR；09-30 晚，P-l2 開跑前） | 寬／深 ResNet 的 block structure：一段層只傳遞主導主成分，容量大於資料集時出現 | P-l2「加寬出現閒置」的先行觀察；可用他們的「容量／資料集大小」作 κ 的解釋變數 | ☐ P-l2 判定時引；☐ L5 補一列 |
| **開跑前對照表**（`prerun-literature-check-2026-09-30.md`） | A28／A27／A29b／A31／A33／P-l2／A32／Zangrando 核算逐項對過 | 沒有一項統計量與格子都被做過；方向都已知 | ☑ |
| **Chen, Bu & Yan, NeurIPS 2024**（`chen2024matthew`；09-30 讀原文） | 每層各通道權重範數變異數的走勢（IS／DS）判「層夠不夠寬」；VGG-16／ResNet-18 CIFAR-10 前後段太寬、中段太窄；流線型寬度同 FLOPs 參數 −40% 準確率 +0.6–2.4；VGG16_bn 初始化 cos(∇, W) ≤ 0.0003（歸因高維） | (i) **兩把互不相干的尺在 VGG-16 指同一處**（後段閒置）——摺入段引為收斂證據；(ii) 他們要到收斂才判、我們初始化就讀——控制器段的差異化；(iii) 0.0003 是 B6 的前作觀察、機制是 BN 不是高維 | ☐ 改稿 L5 段、`sec:a18` 對照一行、B6 段引 0.0003 |
| **Joudaki, Daneshmand & Bach, ICML 2023**（`joudaki2023bridging`）＋ **NeurIPS 2023**（2305.18399；09-30 讀定理） | BN-MLP 初始化：批次 Gram 對均場不動點集中，誤差 e^{−α·depth}＋n/√d（幾何遍歷是猜想）；LN／BN＋非線性激活以 log β_σ 速率把 Gram 推向等距；深層忘掉輸入 | (i) 環 1′ 的非線性有限寬度版本；(ii) **兩個可證偽預測**：忘掉輸入（既有資料：plain／dense 家族成立、ResNet 不成立——恆等捷徑帶輸入）、深層變寬（位置取樣看不到，因相鄰 patch 是他們排除的「重複樣本」）；(iii) 他們的區間 n ≪ d 是我們硬規則 2 排除的 | ☑ 忘掉輸入已用 A7 資料檢（L1 地圖，探索性）；☐ 補 seed 2–3；☐ 環 1 的「假設失效」句 |
| **Garrod & Keating, NeurIPS 2025**（`garrod2025persistence`；09-30 讀主文定理） | 深線性／ReLU CE-UFM：L 大時最優秩 ≤ q_K ≤ 2；DNC 非最優但是 PSD 臨界點、地景維度隨寬度升 → 寬層更常落 DNC | 環 3 全部理論預測秩 ≤ K；**沒有一個預測 κ ≫ K**——B6 是唯一候選解釋，另一候選是對象不同（池化表徵 vs 位置取樣）；第一個寬度相依的塌縮陳述 | ☐ 環 3 段引；☐ A35 判後對照 |
| **Bayat, Behrouz & Courville 2026**（TLM，arXiv 2606.23670）＋ **Mudarisov, Burtsev & State 2026**（2608.02064） | LLM MLP 寬度前寬後窄勝均勻（440M 困惑度 −0.32；後寬 +1.0）、機制 cos(MLP 輸出, 殘差流) 隨深度升；用參考模型一次前向的幾何量定每層 FFN 寬度並固定預算最佳化 | (i) 「寬度不該是超參數」2026 已是 LLM 活題，全是實證或手選——(i) 訓練前可算是空格；(ii) **方向與 CNN 相反**（LLM 前寬、CNN 後寬）要寫成兩種對象；(iii) 他們的 ρ_l 可在 A30 資料上直接算 | ☐ L5 段引並劃界；☐ 探索性：A30 的 ρ_l |
| **Boix-Adserà, Littwin, Abbe, Bengio & Susskind, NeurIPS 2023**（2306.07042；讀摘要與定理陳述） | transformer 的 W_t − W_0 秩隨訓練逐步上升（對角權重＋小初始化下證明；ViT CIFAR 實證） | 環 2 在 transformer 上的「權重端」版本；我們 A30 量的是激活端水準 0.16→0.38 上升——同向、不同對象 | ☐ A30 段一句 |
| **Zhang & Saab, ACHA 2026**（`zhang2026lowRankGuarantees`；讀 §1.3） | 資料相依訓練後低秩壓縮的三個恢復定理（MSE ≲ r(d₁＋d₂)σ²/(d₁d₂)＋ε；近似低秩；ReLU） | (ii) 摺入不掉分有定理可引；誤差 ＝ 丟掉的變異 ＋ 秩×維度噪聲——對應 τ 與「多類無餘裕」；**他們把 r 當給定**，r 由什麼決定是我們的題目 | ☐ (ii) 段一句 |
| **Zheng, Li, Saratchandran & Lucey, NeurIPS 2025**（2505.19985；讀命題 1） | ConvMixer／ViT 的空間混合層可用隨機脈衝濾波器代替訓練：只要 D ≥ k·f²（k＝輸入 patch 嵌入的秩、f 核大小），任何訓練結果可由只訓通道混合層達到；中間輸出的 stable rank ≪ min(N, D) | 「通道數 × 核面積 ≥ 秩」是 r_max 型的計數論證用在初始化設計上；A33 的「初始化方式是輸入」多一個例子（結構化初始化改小資料集表現、ImageNet 持平） | ☐ 環 1 一句（初始化方式）；不引入正文除非 A33 判出效應 |
| **Staats, Thamm & Rosenow, NeurIPS 2025**（`staats2025smallSingular`；讀摘要與圖 2） | 預訓練 transformer 權重的奇異值兩端都偏離 MP 本體；偏離處的奇異向量與**激活共變異數的特徵向量**重疊大；砍掉偏離 MP 的小奇異值比砍本體更傷困惑度 | 尾端有資訊——支持我們用高 τ（0.999）而非 0.95 做摺入、也解釋「多類任務無餘裕」；L4「權重譜 vs 激活譜」的橋（他們直接量重疊） | ☐ (ii) 段或 L4 段一句 |
| **Gahtan, Galil & Bronstein, ACML 2026**（2609.31098；讀摘要與定義） | 殘差流的「有效深度」D_eff＝把層間表徵相似度的自相關剖面聚成一個數；正交更新的閉式參考 2L/(L＋1) < 2；16 個 LLM 中 15 個低於參考 → 更新是相關的、不是深度沒用 | 環 5 的 LLM 端對照：他們量層間相似度、我們量加法處的譜；「校準過的參考值」的做法（先算隨機／正交參考再讀數）與我們的隨機對照同型 | ☐ 環 5 或 Related Work 一句 |
| **Han, Petzka, Adilova, Kleesiek & Kamp, NeurIPS 2025**（2509.17738；讀摘要） | 用 grokking 把泛化與訓練動力在時間上分開：神經塌縮與相對平坦都在泛化開始時出現，但塌縮不是泛化的必要條件、平坦是 | 環 3 的措辭：塌縮是訓練動力的副產品而非目標——支持我們把「終端塌縮」寫成 few-class 訓練到位的性質而非任務需求；引一句劃界 | ☐ III-C 一句 |
| **Dees, Agarwala & Lowman 2022**（2201.11813；讀摘要） | autoencoder Jacobian 特徵值分佈：初始化與訓練 >100 epoch 定性不同，但早期幾個 epoch 就已接近訓練完的分佈；與隨機矩陣乘積理論比較 | 環 2「早期就定型」的又一個（不同對象：Jacobian、MNIST autoencoder）實證；不引入正文 | — |
| **Wan, Zhu, Zhang & Sun, NeurIPS 2021**（`wan2021spherical`；讀定理 1） | 縫放不變層＋SGD＋wd：‖w‖² 線性收斂到 √(Lη/(2λ))、角度更新收斂到 √(2ηλ)，並給出達到平衡的充分條件 | B6 之後「訓練到達哪裡」的精確引用；κ 的第一個定量線索：範數由 (η, λ, 梯度能量) 定 | ☑ 已進理論鏈與草稿 |
| **Galanti, Siegel, Gupte & Poggio 2024 v7**（2206.05794；09-30 讀定理與結論） | 展開 SGD＋wd 的遞推：權重＝記憶窗內 n·B 個秩 ≤ m_l 的梯度（引理 3.2、定理 3.3，不假設駐點）；batch 小／lr 高／wd 大 → 秩低（ResNet-18 CIFAR-10，無 momentum、階梯 lr） | **B7.1 的機制是他們的**；他們的界對卷積是空的（m_l＝patch 數），列空間包含把它變成有內容的；他們列的未來工作（momentum、中間層 NC）＝B7 的兩件事 | ☑ 已進 B7；☐ 改稿 III-C 引並劃界 |
| **Paul, Ganguli & Dziugaite, NeurIPS 2021**（`paul2021dataDiet`）＋ **Toneva et al., ICLR 2019**（`toneva2019forgetting`；讀摘要） | EL2N ＝ ‖softmax−onehot‖ 幾個 epoch 後就能挑出重要樣本、砍半 CIFAR-10 不掉分；遺忘事件集中在少數樣本 | **B7 的 τ 寬度旋鈕**：Γ 的集中權重就是 EL2N；κ ＝ 窗內仍有 EL2N 的樣本 × 遮罩多樣性——A24／A25 的兩個方向都由它讀出，且前向可算 | ☐ gradgram 判後登記 P-b7.5 |

## 還沒查、開工前要先查的

| 主題 | 為什麼 | 先查什麼 |
|---|---|---|
| ~~Th8（ReLU 這一步）~~ | **09-30 已查**：已知（Fan–Wang 2020、Louart–Liao–Couillet 2018），假設在卷積 patch 上不成立 | `notes/th8-prior-art-2026-09-30.md` |
| S2 初始化剖面可算 | B+ 脊椎第二段 | conv 的 mean-field／NNGP 有限寬度修正；Everett 2026 |
| ~~S3 訓練是否重排順序~~ | **09-30 已查**：`notes/s3-prior-art-2026-09-30.md` | 投稿前再查一次 2026 年的 arXiv |
| ~~transformer 的剖面（A30）~~ | **09-30 已查**：`notes/a30-prior-art-2026-09-30.md` | 投稿前再查一次（搜尋不是窮盡的） |
| ~~表徵／權重／梯度三者對齊（CRH／PAH）~~ | **10-02 已查**：`notes/crh-prior-art-2026-10-02.md`（Ziyin et al. ICLR 2025 全文；A32、P-b7.2 是其 forward GWA／RGA 的秩相關版本；B8 檢其冪次形式在 conv–BN 層不成立） | Semantic Scholar 只列 15 篇引用，投稿前用 Google Scholar 再掃一次 |
| 梯度加權的寬度 | 上表的新想法 | Fisher／AGOP 加權的 PCA、task-relevant dimension、活性子空間（active subspaces） |
| 轉移矩陣 M̃ ＝ Γ^{−1/2}Σ_outΓ^{−1/2}（B10） | **10-06 初查**：`notes/b10-prior-art-2026-10-06.md`（手邊 11 篇全文＋arXiv 三個查詢；CRH 是最近鄰，NFA 文獻只有輸入側，Sun 2026 的「神經互動」是輸入側 AGOP 非對角的遠親） | S2 10-06 限流，CRH 被引清單未掃；mutual coherence 文獻未查 |
