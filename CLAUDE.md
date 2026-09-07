# effective-width — 專案規則

卷積網路逐層量測「有效維度 ÷ 可用寬度」。目標是 **IEEE 中上程度期刊，不投
會議**（作者健康因素不便出國）。作者獨立作業，無研究生、無研究經費、不購置
硬體，只租雲端 GPU 且成本要壓低。

**開工前必讀**（依序）：
0. `notes/reframe-2026-09-07.md` —— 現行論文框架（標題／摘要／§I／§II 已依此改寫，§V 待改）
1. `notes/review-2026-09-03.md` —— 對第一輪結果的敵意自我審查。**現在的
   論文框架已被這份審查否掉，不要照 `paper/main.tex` 現有的主張往下寫。**
2. `notes/round2-2026-09-04.md` —— 第二輪（22 組權重）結果的讀法；承重主張
   表的現行數字都來自這裡（`results/checkpoint_analysis.txt`）
3. `notes/analysis-plan.md` —— 預先登記、§7 結果、§8 偏離紀錄
4. `notes/status-zh.md` —— 中文現況與下一步（`README.md` 已改為英文的公開首頁，2026-09-07）

---

## 硬規則

1. **不要用 Marchenko–Pastur 定門檻。** CNN 激活譜是冪律、α ≪ 1，沒有
   bulk edge 可切。這條已經否決過一次。
2. **`n_over_C_ok=False` 的列一律不得報告。** n < C 時樣本共變異數在構造上
   秩虧損，那是取樣假象不是量測。門檻 n/C ≥ 50。
3. **正規化的分母是可達秩 `r_max`，不是標稱通道數 `C`。**
   `r_max = groups · min((C_in/groups)·k_h·k_w, C_out/groups)`。
   ResNet-50 有 20/53 層是 1×1 擴張，`k*/C` 在架構上就不可能超過 0.25。
   **健全性檢查：`k*(0.999)/r_max` 不得超過 1。超過就是 r_max 算錯了**
   —— 這個公式我寫錯過兩次（忘記 kernel 空間範圍、grouped conv 漏乘 g）。
4. **四種「維度」概念要分開寫。** 流形內在維度 ≠ 線性張成的秩 ≠
   participation ratio ≠ 達 X% 變異的主成分數。Ansuini 自己的 VGG-16：
   同一層 PC-ID ≈ 200 而 TwoNN ID ≈ 18。
5. **depthwise 與 dense 卷積不可混在同一條深度曲線上。** depthwise 不混合
   通道，其通道共變異數是不同的對象。ConvNeXt-T 是 18 depthwise + 4 dense，
   兩群中位數差約 0.4。
6. **conv 與 post-activation 輸出分開記錄。** ReLU 使表徵非負，改變共變異數
   結構。

## 統計規則（2026-09-03 審查後新增，最重要）

7. **不得用 `rho_late` 支撐任何主張。** 它是最後 1/3 層的秩相關，對 VGG-16
   只有 **4 個點**（精確 p = 0.167）。全篇 36 個此類檢定、零多重比較校正。
   `analyse.py` 印出的 H1/H2 判定就是由它算的，**改寫前不可引用**。
8. **主張一律用逐層配對檢定**（Wilcoxon／符號檢定／ρ 對深度），用**全部**
   通過門檻的層，不是最後幾層。配對檢定的檢力高一個數量級，而且它是對的。
9. **報告效果幅度時，必須同時報告該曲線自身的層間抖動。** VGG-16 宣稱的
   0.059 塌縮小於它自己的相鄰層階差中位數 0.074。
10. **n = 1 已於 2026-09-04 部分解決。** ResNet-50 有 6 個 ImageNet-1k 配方
    （每層跨配方 SD 0.05，相鄰層抖動 0.09）；其他架構仍是一組權重。**只有
    在六個配方上一致的東西才能寫成 ResNet-50 的性質**；VGG／ResNet-18／
    DenseNet／MobileNet 的任何層級細節仍無誤差棒。
11. **「重現」二字只能用在 CIFAR-10 的 Garg 關卡上。** 2026-09-03 的關卡：
    同資料集、同架構、自己訓練，全 13 層 r = 0.981、MAD 0.052，通過。
    ImageNet 對 CIFAR-10 的第一輪對照（r = 0.125）仍是跨資料集對照，不是重現。
    寫重現時要連 features.27／.30 差 0.2 一起寫。
12. **探索性發現必須標示為探索性**，並與預先登記的部分分開陳述。

## 承重主張的現況（2026-09-04，`results/checkpoint_analysis.txt`）

| | 檢定 | 狀態 |
|---|---|---|
| r_max 是對的分母 | 14/14 有 1×1 擴張的模型：用 C 全 p<0.01，用 r_max 全 p>0.01；上界 1/6、1/4、1/2 三種都成立；22 模型 max k*/r_max ≤ 1.000 | **強烈倖存**。上界本身已知（Han 2021 ReXNet、Kim 2018），我們的是「當量測分母」。stem 層只到上界的 0.3–0.4（RGB patch 本身低秩），要寫成「必要非充分」。見 `notes/rmax-novelty-2026-09-03.md` |
| 池化一致壓低水準 | 687/703 dense、79/80 depthwise（訓練模型）；16 例外中 15 個是 C=32 的整數平手；下降÷自身抖動（r_max 版）19 個模型中 18 個 >1，中位 2.2，DenseNet-121 0.5 | **強烈倖存**；新穎性已查（`notes/pooling-novelty-2026-09-03.md`） |
| conv 輸出 vs block 輸出 | 161/161 個 bottleneck 的 block 輸出超過 conv3 的上界，比值約 4 | **強烈倖存**（新增） |
| trained ≠ random | ResNet-50 六配方全 p ≤ 3e-8；ConvNeXt dense/dw 分開後 p=0.013／0.21 | **ResNet-50 倖存，ConvNeXt 降級**。舊的 p=6.3e-4 混了 dw 與 dense，違反規則 5，不得引用 |
| Garg 驗證關卡 | CIFAR-10 全 13 層 r=0.981 | **通過** |
| 剖面的形狀（駝峰／上升） | 見規則 7 | **不倖存**。六個 ResNet-50 配方的 k*/r_max 剖面順序兩兩 ρ 0.79–0.92 是可以講的（探索性） |
| 訓練對剖面的作用依 block 類型分三種 | 7 個家族、各 2–6 組權重＋3–5 個隨機種子、同一批影像：plain/basic/dense 順序為架構決定（T-R 0.6–0.75）＋水準抬高；bottleneck 順序學出來（T-R −0.08、T-T 0.84）；inverted residual 無跨配方共同剖面（T-T ≈ 0），水準隨配方差 4 倍，depthwise 訓練後減半 | **候選主張**（2026-09-04）。VGG/CIFAR 軌跡：最終剖面與初始化 ρ 0.93、epoch 40 定型（§9）。見 `notes/round2-2026-09-04.md` §7–9 |

論文的標題、H1–H4 與整個 §V 目前都繞著最後一列寫，必須重整。

## 下一步（依優先序）

1. ~~查證 r_max 的文獻新穎性~~ —— 2026-09-03 已查，見上表。**論文不得再寫成
   「發現分母錯了」**，要寫成「文獻忽略了一個已知上界」。上界只在 conv 輸出
   （activation 前、殘差相加前）成立，block 輸出不受限（161/161 已量到）。
2. ~~15–20 個公開 checkpoint 的量測~~ —— 2026-09-03 跑完，22 組權重，
   `notes/round2-2026-09-04.md`。
3. ~~依新排序重寫論文~~ —— 2026-09-07 全文（標題、摘要、§I–§VI、結論）已依 `notes/reframe-2026-09-07.md` 重寫。剩餘 `\todo`：e-mail、Reproducibility 網址。**尚未編譯**（本機無 TeX），下一步是在 Overleaf 或 pod 上編譯一次抓錯。
4. 小補量測（便宜）：~~隨機對照多抽幾個種子~~（2026-09-04 已做，§7–8）；
   ~~MobileNet／EfficientNet 的池化效果改用 r_max 版統計量~~（2026-09-07 已做，
   `round2` §3 補記：/C 比值 0.08–0.17 → /r_max 1.25–2.11；只有 DenseNet-121 <1）。
5. CIFAR 2×2 為選配：`bash code/scripts/run_cifar_2x2.sh`。

---

## 慣例

- **GitHub 遠端（2026-09-07 起）**：`git@github-layerspec:layerspec/effective-width.git`，私人 repo，
  帳號 `layerspec` 是這篇論文專用，與作者私人帳號分開。這個 repo 的 `git config user.*`
  已設成 layerspec 的 noreply 身分，**不要改回全域設定**；SSH 走 `~/.ssh/config` 的
  `github-layerspec` 別名（金鑰 `id_ed25519_layerspec`）。`gh` CLI 登入的是私人帳號，
  對這個 repo 一律用 git 指令，不用 `gh`。歷史已在推送前全部改寫成 layerspec 作者（備份分支已於同日刪除）。

- 表格與圖由資料生成再 `\input`，不手抄。論文現用的三張表與圖 1／圖 5：
  `cd code && python3 scripts/checkpoint_analysis.py --results ../results --latex-models ../paper/table_models.tex --latex-pooling ../paper/table_pooling.tex --latex-families ../paper/table_families.tex > ../results/checkpoint_analysis.txt`
  `python3 scripts/paper_figures.py --results ../results --out ../results/figures`
  （`layerspec.analyse --latex` 產的 `table_shape.tex` 是 rho_late 表，論文已不引用）
- 重算審查數字：`cd code && python3 scripts/review_checks.py --results ../results`
- 承重主張（全部 checkpoint）：`cd code && python3 scripts/checkpoint_analysis.py --results ../results`
- 測試與本機量測：`~/.venvs/effwidth/bin/python`（2026-09-07 建的持久 venv，torch 2.14＋MPS、timm、pytest）。`cd code && ~/.venvs/effwidth/bin/python -m pytest tests/test_core.py -q`。系統 python3 只有 pandas／scipy／matplotlib，夠跑分析與畫圖。
- 長時間本機工作用 `nohup caffeinate -i <script> > log &`，不要放在 session scratchpad 裡（會被清）。目前的隊列：`results/trajectory_queue.sh`（log 同名 .log）
- 論文編譯：本機已裝 tectonic（`brew install tectonic`，2026-09-07），`cd paper && make tectonic` 產生 `main.pdf`（自動抓 IEEEtran）；也可用 Overleaf
- 改動 `hooks.py` 的鍵值方式時要格外小心：ResNet block 會重用同一個
  `nn.ReLU`，只用模組名當鍵會**靜默地**把不同的激活位置合併起來。已用
  call index 修掉，並有測試。
