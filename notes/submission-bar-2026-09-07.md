# 投稿門檻（2026-09-07 定）—— 寧可多花幾個月，不投會被退的版本

作者原話：「既然目標是 TPAMI，就用最高標準做到最好；寧可現在多花幾個月，也不要
投出去等一兩年被退稿。」篇幅與算力（RunPod）不是限制。以下每一項做完才投。

## A. 證據（缺一項都是審稿人的現成理由）

| # | 項目 | 為什麼 | 成本 | 狀態 |
|---|---|---|---|---|
| A1 | ResNet-101／WRN／ResNeXt／MobileNetV2 各 3 種子＋多配方，補進家族表 | 深度與寬度和 block 類型混淆 | Mac，排隊中 | 跑 |
| A2 | 受控 block 實驗：CIFAR-10 同深度 basic vs bottleneck ResNet、各 3 種子；CIFAR MobileNetV2 3 種子；量 R-R／T-T／T-R 與軌跡 | 「依 block 類型」唯一乾淨的檢定 | RunPod ≈ US$20 | 未開始，**先寫進 analysis-plan §9 再跑** |
| A3 | inverted residual 同配方多種子（自己訓練 MobileNetV3-L／EfficientNet-B0 ImageNet，各 3 種子） | 「現有 checkpoint 不共享剖面」升級為可判定主張的唯一方法 | RunPod ≈ US$60–100 | 未開始；先看 A1 與四個 tf_ 權重的結果再決定 |
| A4 | 分解量測（Σ_out = W Σ_patch Wᵀ）跑滿 22 組權重＋隨機種子 | §VI 新節的證據 | Mac，排隊中 | 跑 |
| A5 | 正交正則化介入：CIFAR VGG-16／ResNet-18，無／SO／SRIP／ONI 各 3 種子，逐層對照 A4 的反事實預測 | 「理論預測 → 實驗命中」是這篇唯一能放的理論味 | RunPod ≈ US$20 | 未開始，先預先登記 |
| A6 | 子集結果在全驗證集重跑一次（六配方五種子、七家族） | 消除「6,400 張子集」這個現成質疑 | RunPod ≈ US$5 | 未開始 |
| A7 | 多個影像子集的 bootstrap，給每個承重數字信賴區間 | 現在只有跨配方 SD，沒有抽樣 CI | Mac | 未開始 |
| A8 | 第二個驗證關卡：對 Elmoznino & Bonner 的 ResNet-50 block 輸出 PR 值 | 只對 Garg 一張表驗證，審稿人會要第二個 | Mac（block hook 已有） | 未開始，要先取得他們的逐層數字 |
| A9 | 軌跡：ResNet-50 CIFAR＋VGG 三種子 | §V.J 的時間面 | Mac，跑 | 跑 |
| A10 | 投影錨點推到第二個架構（VGG-16、ResNet-18）與全驗證集 | 一個模型一個子集不夠 | Mac | 未開始 |

## B. 論文本身

- B1 兩位獨立審稿代理再各做一輪（數字審計＋論證審計），在所有資料到齊之後。
- B2 全文英文潤稿一輪（句長、被動、術語先定義後使用）。
- B3 每個表和圖由腳本產生、每個數字能追到 `checkpoint_analysis.txt` 的一行。
- B4 analysis-plan.md 翻成英文，偏離紀錄補到投稿當天。
- B5 §II 五條新 bib（Bansal、Massart、Huang×2、Wang）逐條核對原文；Massart 要看 IEEE 版。
- B6 標題與摘要在全部結果出來後最後定。

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
| T1 | 命題 1（單層傳遞）：Ostrowski 型不等式夾住 λ_i(WΣWᵀ)，分 C_out ≥ d／< d；等距核時 k*_out = min(k*_patch, r_max) | 未寫 |
| T2 | 命題 2（正交化反事實）：W_o Σ W_oᵀ 的譜 = Σ 在 W 列空間壓縮的譜；Table VI 的 ortho 欄由此定義 | 未寫 |
| T3 | 命題 3（跨層遞迴）：Σ_patch^(ℓ+1) 是 post-activation 空間共變異數的線性像（unfold），給沿深度的上界；連到 Feng 2022 | 未寫 |
| T4 | 用現有十組分解資料逐層驗證 T1–T3 的不等式全部成立（對證明的檢查） | 未做 |
| I1 | `layerspec.profile(model, loader)` API：k*(τ)/r_max、閘門、W／patch 分解、投影檢查；文件、測試、PyPI | 未做 |
| I2 | 應用段：k*(0.999) 子空間投影 = 逐層低秩分解 W = V_k(V_kᵀW)，算 FLOPs 節省；引 Zhang et al. 2015 TPAMI（回應 PCA 低秩分解）及後續，定位為量測驅動的秩選擇、免微調、配方相依 | 未做 |
| M1 | §I 前三段改以三個做不到的事開場：可比的量測、數字的功能意義與來源、正交正則化的白化假設無工具可檢 | 未做 |
| W1 | 附錄「主張→證據→狀態→表格行」總表 | 未做 |
| W2 | 最後一輪「一句話貢獻」測試（沒看過稿的代理） | 未做 |
| C4 | §I 末段明寫影響：跨論文可比的尺、bottleneck 誤讀的糾正、瓶頸從資料到核、對正交正則化的可檢驗預測 | 未做 |

順序：T1–T4（不需新資料）→ I2 → M1／C4 → 資料齊後 I1、W1、W2。
