# effective-width

研究專案：**卷積網路的名目寬度，究竟用掉了多少？**

兩篇論文共用一條主線，加上一條支線。目標是 IEEE 中上程度期刊，不投會議。

---

> **2026-09-03 稍晚：第一輪結果經敵意自我審查後，論文的框架被否掉。**
> 量測管線是乾淨的，但形狀相關的主張都不成立（`rho_late` 對 VGG 只有 4 個
> 點）。開工前請先讀 `CLAUDE.md` 與 `notes/review-2026-09-03.md`。
> 下方「第一輪結果」一節已依審查修訂。

## 現況（2026-09-04）

| | 狀態 |
|---|---|
| 第一篇 · 有效寬度量測協定 | 兩輪量測完成（22 組權重＋本機 7 家族多種子）。論文全文已依新框架重寫（2026-09-07），待編譯；剩 2 個 TODO |
| 第二篇 · 非調參的寬度判準 | 規劃中，接續第一篇，共用量測基礎設施 |
| 支線 · 正交正則化受控再評估 | 完整實驗設計已寫，約 1,200 GPU-hours，時程未定 |

### 第一輪結果（詳見 `results/analysis.txt`、`notes/analysis-plan.md` §7）

- **倖存的 H2 是水準差，不是形狀差。** 全域池化在 **108/108 層**上一致地
  壓低量測值，無一例外。但「池化改變了形狀」用正確的配對檢定
  （ρ(深度, 池化−抽樣)）四個架構全部不顯著，VGG 的符號還是反的
  （−0.352, p = 0.24）。
- **所有形狀判定不成立。** `rho_late` 對 VGG-16 是 4 個點的秩相關
  （精確 p = 0.167），全篇 36 個此類檢定零校正。宣稱的 0.059 塌縮小於該
  曲線自身的相鄰層抖動 0.074。
- **trained ≠ random 倖存**（Wilcoxon p = 6.3e-4 / 4.2e-6）；
  **H5 乾淨**（n/C 100→250 變化中位數 0.0000）。**H3 待 CIFAR 2×2**。
- **Garg 對照不是重現。** 全 13 層 r = 0.125；先前引用的 0.968 是事後挑的
  前 8 層，而且我們跑 ImageNet、他們跑 CIFAR-10。
- **方法上的修正：正規化分母應該是可達秩 r_max，不是標稱通道數 C。**
  ResNet-50 有 20/53 層是 1×1 擴張，k\*/C 在架構上就不可能超過 0.25。
  改用 r_max 後中位數 0.250 → 0.422，與 VGG（0.422）、ResNet-18（0.533）同級，
  「寬層只用一成寬度」那個說法是分母造成的假象，已撤回。

### 第二輪結果（2026-09-03 跑，2026-09-04 分析；詳見 `notes/round2-2026-09-04.md`）

22 組權重（含 6 個 ResNet-50 配方）、Garg 關卡、block 輸出。摘要：
r_max 分母 14/14 模型成立；池化壓低 687/703；block 輸出超過 conv3 上界
161/161；Garg 關卡 r=0.981 通過；trained≠random 只在 ResNet-50 成立，ConvNeXt
分開 depthwise 後不顯著。數字由 `code/scripts/checkpoint_analysis.py` 產生。

### 下一步

0. ~~查證 r_max 的文獻新穎性~~ —— 已查（2026-09-03），見 `notes/rmax-novelty-2026-09-03.md`
0. ~~量 15–20 個公開 checkpoint~~ —— 已跑（2026-09-03），見 `notes/round2-2026-09-04.md`
1. ~~**依新框架重寫論文**~~ —— 2026-09-07 全文重寫完成（`notes/reframe-2026-09-07.md`）。
   定位是「量測協定＋三個陷阱＋訓練作用依 block 類型分三種」。目標期刊 TPAMI。
   **下一步：編譯一次**（Overleaf／pod），補 e-mail 與程式碼網址。
2. ~~小補量測：隨機對照多個種子；MobileNet 系列池化效果用 r_max 版~~（2026-09-04／09-07 已做）。
3. CIFAR 2×2（約 US$4，選配）—— `bash code/scripts/run_cifar_2x2.sh`
4. 取得 IEEEtran.cls（Overleaf 或 texlive-publishers）

跑法（機器上）：

```
hf auth login
python code/scripts/fetch_imagenet_val.py --out /data/imagenet_val
bash code/scripts/run_on_rented_gpu.sh /data/imagenet_val
```

**不需要 devkit，也不需要 valprep 重整目錄。** 管線只算激活共變異數、從不讀標籤，
所以接受扁平目錄，驗證集 tar 解開後直接指過去就行。

---

## 主線的兩篇

**第一篇 — How Much of a Convolutional Layer's Width Is Actually Used?**

逐層量出「有效維度 ÷ 名目寬度」。四點貢獻：

1. 調和矛盾 — Garg et al.（IEEE Access 2019）量到駝峰，Elmoznino & Bonner（PLOS CB 2024）量到單調上升。在同一批模型的同一批層上同時算兩種指標。
2. 補上沒人量過的架構 — Garg 明確排除 ResNet（因為 shortcut），ConvNeXt 晚於全部相關工作。
3. 初始化對照 — 同架構未訓練的曲線，分離「學到的結構」與「架構加輸入統計本身就有的結構」。文獻裡沒人報過。
4. 取樣量紀律 — n < C 時樣本共變異數在構造上秩虧損，會無聲地製造低秩假象。若部分既有曲線是取樣量假象，那本身就是發現。

**第二篇 — 非調參的寬度判準**

用 Horn 的 parallel analysis（置換建虛無分布，免疫於重尾）或 Minka 貝氏證據，取代所有現有方法那個調出來的門檻。對照 Self-Expanding NN 與 splitting steepest descent。需要訓練，約 1,200 GPU-hours。

**支線 — 正交正則化受控再評估**

這個假設沿用八年，2024–2026 沒有任何一篇專門檢驗它是否幫助泛化。四個對照臂（Ctrl-Random / Ctrl-Norm / Ctrl-Spectral / Ctrl-Early）沒有人跑過。

---

## 資料夾

```
code/       layerspec 量測管線（見 code/README.md）
results/    每次跑的原始 CSV 與 npz
            結果圖在 results/figures/（PDF 給論文、PNG 給人看）
paper/      草稿、bib、IEEE 模板
refs/       要親自確認的 PDF
notes/      零散想法
.snapshots/ 程式碼快照
```

## 線上文件

- **正交萃取文獻地圖** — 71 篇，按「已被佔走 / 鄰近必引 / 對你有利」分類
- **正交正則化再評估** — 支線的完整實驗設計

兩份都在 claude.ai/code/artifacts。

## 必讀的三條紀律

1. **不要用 Marchenko–Pastur 門檻。** CNN 激活譜是冪律、α ≪ 1，沒有 bulk edge 可切。
2. **樣本數必須遠大於 C。** 任何 `n_over_C_ok=False` 的列都是假象，不是量測。
3. **四種維度概念要分開寫。** 流形內在維度 ≠ 線性張成的秩 ≠ participation ratio ≠ 達 X% 變異的主成分數。Ansuini 自己的 VGG-16 數字：同一層 PC-ID ≈ 200 而 TwoNN ID ≈ 18。

## 已查證的三篇（全文已讀，2026-09-03）

詳見 `refs/verification-notes.md`。三句話：

1. **NORTH\*** — 有做卷積（VGG-11、WRN-28 於 CIFAR），而且它的觸發統計量
   （後激活的 ε-數值秩 ÷ 層寬）和第一篇量的 k\*/C 是同一家族。
   **必須引在 Method 一節**，不能只當成一個成長方法帶過。第一篇的貢獻仍在——
   他們沒做剖面、沒做 ImageNet、沒做預訓練模型、沒做初始化對照。
   作者自承仍需人為設定層寬上限，這是第二篇的開口。
2. **GOR** — 命名衝突屬實，Eq. 2 是逐組 Gram 懲罰（組內正交），全文無跨組項。
   注意他們自己的 inter/intra 術語也會誤導，引用時直接寫式子。
3. **PCA-AE** — 序貫凍結確認為真；但**沒有** PCA 等價定理（更正先前說法），
   且只作用在 bottleneck、d_max 由使用者指定。
