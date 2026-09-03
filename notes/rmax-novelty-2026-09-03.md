# r_max 新穎性查證（2026-09-03）

對應 `review-2026-09-03.md` §8 與 CLAUDE.md「下一步 1」。問題：「1×1 擴張層的
通道共變異數秩上界為 C_in」這個觀察，CNN 文獻裡有沒有人講過？

**結論：線性代數事實本身不新，至少兩篇 CNN 論文明講過；但「拿它當逐層有效
維度量測的正規化分母，並證明 ResNet-50 的兩群假象由此消失」沒人做過。**
第一承重點降級為「方法上的修正」，不是「發現」。`paper/main.tex` 第 325–335
行陳述 r_max 時目前**零引用**，投稿前必須補。

---

## 1. 直接先行文獻（必引）

### Han, Yun, Heo, Yoo — ReXNet, CVPR 2021（arXiv 2007.00992）

「Rethinking Channel Dimensions for Efficient Model Design」。最接近的一篇，
而且是 CVPR 正式論文、timm 內建模型，效率架構圈的審稿人一定知道。

- §3.2 原文：「The rank is originally bounded to the input dimension, but the
  subsequent nonlinear function will increase the rank above the input
  dimension [1, 58].」（[1] Amini et al. 2011；[58] Yang et al. 2018 softmax
  bottleneck）
- Fig. 1：**rank ratio（rank / 輸出通道數）對 dimension ratio（C_in / C_out）**，
  對 (a) 單一 1×1 conv、(b) 3×3 conv、(c) inverted bottleneck、(d) depthwise
  IB 各畫一條，1,000 個隨機尺寸網路平均。這張圖就是我們 k\*/C 對 r_max/C 的
  隨機網路版。
- 觀察 (i)：「Drastic channel expansion harms the rank.」
- §2 明點名：「Both of the inverted bottleneck [47] and bottleneck block [16]
  have the convolutional expansion layer with the predefined expansion ratio
  (mostly 6 or 4).」—— 也就是 ResNet 的 4× 擴張。
- 用法是**設計準則**：第一個 1×1 的 expansion ratio ≤ 6、DR ≥ 1/6。
- 量測條件：**隨機權重**，f(WX) 含 BN + 非線性，X ∈ R^{d_in×N}，N > d_out > d_in。
  §6 才碰訓練後模型，但只看**最終特徵**的 nuclear norm（ImageNet val）。

與我們的差異（可寫進論文的區隔）：
1. 他們量非線性**之後**的秩，重點是「非線性能把秩撐回去多少」；我們量 conv
   輸出（activation 之前），上界嚴格成立，且訓練後的擴張層達到上界的 ~93%。
2. 他們是隨機網路的設計啟發；我們是訓練後 ImageNet 網路的量測工具。
3. 他們沒有寫出含 kernel 範圍的一般式（Kim et al. 有，見下）。

### Kim, Yoon, Jeong, Lee — Rank-1 CNN, arXiv 1808.04303（2018，預印本）

Eq. (28)–(37)：把多通道卷積寫成 Y = H[X] W（Hankel/im2col 形式，
Y 為 位置 × 輸出通道），得 rank(Y) ≤ min{rank H[X], N, q}，並明說一般卷積的
上界是 **min{N·d1·d2, q}**，即 min(C_in·k_h·k_w, C_out)。**含 kernel 範圍的那一
項已有人寫過**。能見度低（預印本），但它存在。

## 2. 同一觀念的其他脈絡（相關工作段落用）

| 文獻 | 講了什麼 | 對我們的關係 |
|---|---|---|
| Yang et al., ICLR 2018（softmax bottleneck） | logit 矩陣秩 ≤ 隱藏維度 d | ReXNet 引用的源頭 |
| Bhojanapalli et al., ICML 2020 | head size < 序列長度 ⇒ attention 秩瓶頸 | Transformer 版本的同一件事 |
| Jacot, ICLR 2023（bottleneck rank） | 深網路表示成本 → 非線性函數的秩概念 | 理論；秩受最窄層限制 |
| Baker et al., arXiv 2402.06751（2024） | 梯度秩 ≤ min(rank Z_{i-1}, rank Δ_i)；卷積含 stride/kernel 的顯式式子 | 梯度不是激活，但形式相同 |
| Boix-Adsera, arXiv 2501.19149（2025） | 無限深 ResNet 的 bottleneck rank 來自 embedding/unembedding | 理論 |
| Feng et al., NeurIPS 2022 | Jacobian 部分秩隨深度單調下降 | 無寬度正規化、無 bottleneck 分析（main.tex 已正確區隔） |

## 3. 量測文獻確認**沒有**處理這件事

- Elmoznino & Bonner, PLOS CB 2024：568 個卷積層、ResNet-50 取 16 個 block 輸出、
  10,000 張 ImageNet val。承認「architecture-specific factors can affect ED in
  ways that are independent of learning」，但不正規化，也不談 1×1／bottleneck。
- Ansuini et al., NeurIPS 2019：取 block 輸出。
- HRank, CVPR 2020：單一通道 H×W 的空間秩，不同對象。
- Semantic Scholar 上 ReXNet 的 115 篇引用（抓到的部分）：無一篇把該上界用於
  維度量測的正規化。

**注意**：block 輸出是殘差相加**之後**，skip path 會把秩加回去，上界對它不成立。
所以 Ansuini／Elmoznino 量 block 輸出時只是部分避開；上界咬到的是 conv 輸出
本身，也就是我們量的東西。論文陳述 r_max 時要把「conv 輸出、activation 之前、
殘差相加之前」講清楚，否則會被 ReXNet 的觀察 (ii)（非線性撐高秩）打。

## 4. 對論文的影響

**降級後的第一承重點**：
> 一個已知的秩上界（Han et al. 2021；Kim et al. 2018）從未被有效維度量測文獻
> 納入分母；納入之後，ResNet-50 的受限／未受限兩群差異（p = 1.4e-9）完全消失
> （p = 0.49），且訓練後的擴張層達到上界的 93%。

還算得上貢獻的部分：
1. 把上界當**量測分母**，並在訓練後 ImageNet 網路上驗證 —— 沒人做過。
2. ~~含 groups 的一般式~~ —— **撤回。** g·min((C_in/g)·k_h·k_w, C_out/g)
   = min(C_in·k_h·k_w, C_out)，g 在代數上消掉了，公式與 Kim et al. 的完全相同。
   CLAUDE.md 規則 3 的寫法沒錯，只是不比 Kim 多任何東西；論文可保留 g 形式
   （對 grouped/depthwise 讀者較清楚），但不得列為貢獻，也不需要找 grouped 1×1
   擴張層來驗證。
3. k\*(0.999)/r_max ≤ 1 作為管線健全性檢查。
4. ReXNet 擔心的「劇烈擴張傷害秩」，在訓練後 ResNet-50 上可被上界完全解釋 ——
   這是對 ReXNet 的一個小回應。

**必做**：
- ~~`paper/refs.bib` 加 `han2021rexnet`、`kim2018rank1`；main.tex Eq. (rmax) 前後
  引用；相關工作加一段「秩瓶頸」~~ —— 2026-09-03 已做（未編譯驗證，本機無 TeX）。
- 標題／§I 不得再暗示「發現分母錯了」；措辭改為「文獻忽略了一個已知上界」。

## 5. 查證的侷限

- 只用 WebSearch／arXiv／Semantic Scholar（後者頻繁 429，引用清單可能不完整）。
  沒有查 Google Scholar 的「被引用」全清單。
- 剪枝文獻裡 HRank 的後續工作有可能在 ResNet-50 上注意到 1×1 擴張層秩偏低，
  搜尋沒找到，但不能排除。
- 建議投稿前再做一次 Google Scholar：`"ReXNet" rank "input channel"` 與
  `"expansion" "rank" "bottleneck" ResNet-50 feature`。
