# 投稿門檻（2026-09-07 定）—— 寧可多花幾個月，不投會被退的版本

作者原話：「既然目標是 TPAMI，就用最高標準做到最好；寧可現在多花幾個月，也不要
投出去等一兩年被退稿。」篇幅與算力（RunPod）不是限制。以下每一項做完才投。

## A. 證據（缺一項都是審稿人的現成理由）

| # | 項目 | 為什麼 | 成本 | 狀態 |
|---|---|---|---|---|
| A1 | ResNet-101／WRN／ResNeXt／MobileNetV2 各 3 種子＋多配方，補進家族表 | 深度與寬度和 block 類型混淆 | Mac | ✔ 09-08 |
| A2 | 受控 block 實驗：CIFAR-10 同深度 basic vs bottleneck ResNet、各 3 種子；CIFAR MobileNetV2 3 種子；量 R-R／T-T／T-R 與軌跡 | 「依 block 類型」唯一乾淨的檢定 | RunPod ≈ US$20 | ✔ 09-08 跑完、09-09 判定：**H9.1 不成立**（block 不是原因）、H9.2 成立；plan §10.1，論文 §V.J 已依預登記後果改寫 |
| A3 | inverted residual 同配方多種子（自己訓練 MobileNetV3-L／EfficientNet-B0 ImageNet，各 3 種子） | 「現有 checkpoint 不共享剖面」升級為可判定主張的唯一方法 | RunPod ≈ US$60–100 | 未開始；先看 A1 與四個 tf_ 權重的結果再決定 |
| A4 | 分解量測（Σ_out = W Σ_patch Wᵀ）十組權重（含隨機） | §VI 新節的證據 | Mac | ✔ 09-08（22 組全跑待排） |
| A5 | 正交正則化介入：CIFAR VGG-16／ResNet-18，無／SO／SRIP／ONI 各 3 種子，逐層對照 A4 的反事實預測 | 「理論預測 → 實驗命中」是這篇唯一能放的理論味 | RunPod ≈ US$20 | ✔ 09-08 跑完、09-09 判定：**P9.3 不成立**（實現預測增益一半，核未等距）、P9.4 以排序形式成立、P9.5 成立；ONI 未跑列限制；plan §10.2，論文 §V-L 已寫 |
| A6 | 子集結果在全驗證集重跑一次（六配方五種子、七家族） | 消除「6,400 張子集」這個現成質疑 | RunPod ≈ US$5 | 未開始 |
| A7 | 多個影像子集的 bootstrap，給每個承重數字信賴區間 | 現在只有跨配方 SD，沒有抽樣 CI | Mac | 未開始 |
| A8 | 第二個驗證關卡：對 Elmoznino & Bonner 的 ResNet-50 block 輸出 PR 值 | 只對 Garg 一張表驗證 | Mac | ✔ 09-08（ρ 0.997） |
| A9 | 軌跡：ResNet-50 CIFAR＋VGG 三種子 | §V.J 的時間面 | Mac，跑 | 跑 |
| A10 | 投影校準：八組（六配方＋R18＋VGG）、McNemar、A1 逐層留一 | 一個模型一個子集不夠 | Mac | ✔ 09-08 |
| A17 | **ImageNet 受控 block 實驗**：§9.1 三架構在 ImageNet-1k 同配方各 3 種子（plan §9.4） | 回答「ResNet-50 為什麼重排」：資料集 vs 配方 vs block；把第三輪的否定升級為指認原因 | RunPod A100 ≈ 160 h、US$250–350 | 已預登記 09-09，未跑 |
| A18 | **用尺定寬度再重訓**：CIFAR VGG-16_BN，全寬／尺定寬 τ=0.999／均勻縮窄同參數／尺定寬 τ=0.95，各 3 種子（plan §9.5） | 「尺能指導設計決定」的直接檢驗，回答「所以呢」 | 3090 ≈ 9 h＋分解 3 h、US$10 | 已預登記 09-09；`--widths`／`--save-checkpoints`／`width_from_ruler.py`／`run_a18_pod.sh`／`decompose_a18.sh` 都好了，`results/a18/widths.json` 已算（ruler 34.5% 參數）；**待開 pod 跑**（running-guide A18 節） |

## B. 論文本身

- B1 兩位獨立審稿代理再各做一輪（數字審計＋論證審計），在所有資料到齊之後。
- B2 全文英文潤稿一輪（句長、被動、術語先定義後使用）。
- B3 每個表和圖由腳本產生、每個數字能追到 `checkpoint_analysis.txt` 的一行。
- B4 analysis-plan.md 翻成英文，偏離紀錄補到投稿當天。
- B5 §II 五條新 bib（Bansal、Massart、Huang×2、Wang）逐條核對原文；Massart 要看 IEEE 版。
- B6 標題與摘要在全部結果出來後最後定。
- B7 **頁數**：compsoc 樣式下目前 17 頁（2026-09-08）。投稿前查 TPAMI 當期的頁數規定（一般 14 頁內免費、超頁收費、有上限），必要時把三張 table* 與部分圖移附錄或補充材料。
- B8 已套用 TPAMI 樣式：`\documentclass[10pt,journal,compsoc]{IEEEtran}`、`\IEEEtitleabstractindextext`、`\IEEEcompsocitemizethanks`、`\IEEEraisesectionheading`（2026-09-08）。

## C. 可重現性包

- C1 repo 公開前清掉 `output/`、本機路徑、中文工作筆記（或移到 `notes/zh/`）。
- C2 一鍵重現：`make reproduce` 從 `results/` 重算全部表圖與 PDF；環境檔（requirements 鎖版本）。
- C3 release tag，論文引用該 tag。

## 時程估計

A1、A4、A9 本週跑完；A2、A5、A6 一趟 RunPod（一兩天，US$50 內）；A3 視 A1 結果；
A7、A8、A10 各一兩天 Mac。加 B、C，約兩到三個月。

## D. 2026-09-08 作者再次定標準後新增：理論、實作、動機、影響

作者原話：「動機要確實可以補上目前技術上的缺失、理論要夠扎實、要有真的可以套用的
實作、內文寫作要有邏輯跟嚴謹、要真的對世界有貢獻跟影響。」

| # | 項目 | 狀態 |
|---|---|---|
| T1 ✔ | 命題 1（單層傳遞）：Ostrowski 型不等式夾住 λ_i(WΣWᵀ)，分 C_out ≥ d／< d；等距核時 k*_out = min(k*_patch, r_max) | 未寫 |
| T2 ✔ | 命題 2（正交化反事實）：W_o Σ W_oᵀ 的譜 = Σ 在 W 列空間壓縮的譜；Table VI 的 ortho 欄由此定義 | 未寫 |
| T3 ✔ | 命題 3（跨層遞迴）：Σ_patch^(ℓ+1) 是 post-activation 空間共變異數的線性像（unfold），給沿深度的上界；連到 Feng 2022 | 未寫 |
| T4 | 十組分解資料逐層驗證：Ostrowski 464/465、推論 465/465 | ✔ 09-08 |
| I1 | `layerspec.profile(model, loader)` API：k*(τ)/r_max、閘門、W／patch 分解、投影檢查；文件、測試、PyPI | ✔ 09-09：layerspec 0.2（profile／decompose API、CLI、pyproject、測試 27 個、ResNet-50 回歸逐層一致）；PyPI 上傳待作者註冊帳號後 `python -m twine upload dist/*` |
| I2 | ~~應用段~~ → 降級為一句話：k*(0.999) 投影 = W = V_k(V_kᵀW) 分解，但 conv FLOPs 只省 2–5%（1×1 層 d<C_out 分解反而變貴；§10b），**不是壓縮方法**，只是解釋性錨點；仍引 Zhang 2015 TPAMI 定位 | FLOPs 已算，待寫一句 |
| M1 | §I 以「所有可靠的 k 都是事後的」缺口開場（限定版三句）＋三件要先解決的事 | ✔ 09-08 |
| W1 | 附錄 B 主張總表（Table 7） | ✔ 09-08 |
| W2 | 最後一輪「一句話貢獻」測試（沒看過稿的代理） | 未做 |
| C4 | §I 末段影響陳述（a ruler and three facts） | ✔ 09-08 |

順序：T1–T4（不需新資料）→ I2 → M1／C4 → 資料齊後 I1、W1、W2。

## E. 2026-09-08 作者問「第二篇要併嗎、深度要談嗎」的決定

- 第二篇（非調參寬度判準）**不併**：方法論文需要訓練實驗，算力不允許；本篇只在 §VI 加一段
  「寬度判準應以什麼為目標」（τ=0.999 才是用到的寬度；r_max 分母；正則化能否推高由 A5 決定）。
- 深度**不另開主題**，但補一段零成本的量測：161 個 bottleneck block 的輸出剖面（block 層級、
  Ansuini／Elmoznino 量的張量），與 conv3 上界並排；命題 3 與軌跡的趨勢反轉已涵蓋其餘。
  「有效深度」（Veit 2016 那條線）放 §VI 一段「本協定接下來可量什麼」。

| # | 項目 | 狀態 |
|---|---|---|
| D1 | block 輸出深度剖面（Fig. 7） | ✔ 09-08 |
| D2 | §VI 寬度判準目標段 | ✔ 09-08 |
| D3 | §VI 有效深度留待另篇 | ✔ 09-08 |

## F. 2026-09-08 作者問「原創性、主流資料集、嚴苛消融」後新增

| # | 項目 | 為什麼 | 成本 | 狀態 |
|---|---|---|---|---|
| A11 ✔ | COCO 偵測／分割 backbone（torchvision Faster／Mask R-CNN R50-FPN）在 COCO val 影像上的剖面、r_max、池化、分解 | 只有分類網路是明顯缺口；不需標籤 | Mac 或 pod 1 h | 未做 |
| A12 ✔ | **BN 消融**：conv 輸出（BN 前，本文量測點）vs BN 後的 k*/r_max 剖面與各主張 | BN 是逐通道仿射，不改秩但改 k*；審稿人必問 | Mac，hooks 加 kind="bn" | 未做 |
| A13 ✔ | 取樣位置數消融：16 vs 4 vs 全部位置 | 位置相關性影響有效樣本數 | Mac | 未做 |
| A14 ✔ | 前處理消融：center crop 256→224 vs Resize(224,224) 直接縮放 | 與 E&B 的 6.5% 差異來源 | Mac | 未做 |
| A15 ✔ | 雜訊輸入：高斯雜訊影像上的剖面（訓練與隨機網路） | 把資料因子換成純架構，與隨機初始化互補 | Mac | 未做 |
| A16 | 數值秩容忍度消融（1e-4／1e-6／1e-8）對命題 1 檢查與 κ 的影響 | 命題 1 的檢查依賴它 | 分析 | 未做 |

方法論原創性的誠實定位（寫在 §I 末段）：綜合與校準＋新量測結果（r_max 進分母、估計量配對、
張量量化、分解反事實、跨家族初始化比較、瓶頸從資料到核），不是新演算法；零件各歸原主。

## G. 2026-09-08 作者問「文獻整理做過嗎」

已做的是逐主張的新穎性查證（rmax、pooling、orthogonality、lowrank、width-theory、eb-validation、
bib-check 七份筆記），不是系統性回顧。新增：

| # | 項目 | 狀態 |
|---|---|---|
| B9 | 系統性掃描「卷積網路逐層表徵秩／有效維度」2014–2026，產出比較表（張量、估計量、統計量、分母、初始化對照、形狀），進 §II；必補 tunnel effect（Masarczyk 2023）、RankMe（Garrido 2023）、α-ReQ、low-rank simplicity bias（Huh 2023）、逐層 neural collapse | ✔ 09-08：Table 1（19 篇）進 §II-B；tunnel／Cohen／Schulte／Feng／Chun／Daneshmand／Huh／Stephenson／Harun 已引並回應；39 條 bib 已於 09-08 核對、09-09 套用（`notes/bib-check-2-2026-09-08.md`；仍缺四篇 2020 前 NeurIPS 頁碼、Horn & Johnson 定理號待查書） |
