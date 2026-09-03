# effective-width — 專案規則

卷積網路逐層量測「有效維度 ÷ 可用寬度」。目標是 **IEEE 中上程度期刊，不投
會議**（作者健康因素不便出國）。作者獨立作業，無研究生、無研究經費、不購置
硬體，只租雲端 GPU 且成本要壓低。

**開工前必讀**（依序）：
1. `notes/review-2026-09-03.md` —— 對第一輪結果的敵意自我審查。**現在的
   論文框架已被這份審查否掉，不要照 `paper/main.tex` 現有的主張往下寫。**
2. `notes/analysis-plan.md` —— 預先登記、§7 結果、§8 偏離紀錄
3. `README.md` —— 現況與下一步

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
10. **n = 1。** 每個架構只有一組公開權重，沒有任何誤差棒。**任何關於「形狀」
    的主張在取得跨 checkpoint 分佈之前都不能寫進論文。** 修法很便宜：
    torchvision 的 ResNet-50 有 V1/V2 兩套權重，timm 上有幾十個 ImageNet
    checkpoint，量測本身近乎免費。
11. **「重現」二字不得使用。** `reproduce_garg.py` 至今沒跑過，管線對任何
    已發表結果都尚未通過驗證。與 Garg 的比較是 ImageNet 對 CIFAR-10 的
    **跨資料集對照**，且全 13 層 r = 0.125（只有事後挑的前 8 層是 0.968）。
12. **探索性發現必須標示為探索性**，並與預先登記的部分分開陳述。

## 承重主張的現況

| | 檢定 | 狀態 |
|---|---|---|
| r_max 是對的分母 | 用 C p=1.4e-9；用 r_max p=0.49 | **強烈倖存**（新穎性未查證） |
| 池化一致壓低水準 | 108/108 層 | **強烈倖存** |
| trained ≠ random | Wilcoxon p=6.3e-4 / 4.2e-6 | **倖存** |
| 剖面的形狀（駝峰／上升） | 見規則 7 | **不倖存** |

論文的標題、H1–H4 與整個 §V 目前都繞著最後一列寫，必須重整。

## 下一步（依優先序）

1. **查證 r_max 的文獻新穎性。** 「1×1 擴張的秩受 C_in 限制」在線性代數上
   很淺顯；Transformer 那邊 Bhojanapalli 等人講過 attention head 的 rank
   bottleneck，CNN 這邊很可能有人講過。**這決定了第一承重點還在不在。**
2. 15–20 個公開 checkpoint 的量測（約 US$3–5），解決 n=1
3. CIFAR 2×2：`bash code/scripts/run_cifar_2x2.sh`（分辨類別數 vs 分類頭深度，
   同時補上真正的驗證關卡）
4. 依新排序重寫 §I／§II 框架

---

## 慣例

- 表格由資料生成再 `\input`，不手抄：
  `python3 -m layerspec.analyse --results ../results --latex ../paper/table_shape.tex`
- 重算審查數字：`cd code && python3 scripts/review_checks.py --results ../results`
- 測試：`cd code && python3 -m pytest tests/test_core.py -q`（需要 torch）
- 論文需要 IEEEtran（`texlive-publishers`，或用 Overleaf）
- 改動 `hooks.py` 的鍵值方式時要格外小心：ResNet block 會重用同一個
  `nn.ReLU`，只用模組名當鍵會**靜默地**把不同的激活位置合併起來。已用
  call index 修掉，並有測試。
