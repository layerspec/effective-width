# effective-width

研究專案：**卷積網路的名目寬度，究竟用掉了多少？**

兩篇論文共用一條主線，加上一條支線。目標是 IEEE 中上程度期刊，不投會議。

---

## 現況（2026-09-03）

| | 狀態 |
|---|---|
| 第一篇 · k*/C 量測 | 管線寫完，7/7 測試通過。**等 ImageNet 驗證集** |
| 第二篇 · 非調參的寬度判準 | 規劃中，接續第一篇，共用量測基礎設施 |
| 支線 · 正交正則化受控再評估 | 完整實驗設計已寫，約 1,200 GPU-hours，時程未定 |

### 下一步

1. 到 image-net.org 註冊、下載驗證集（6.7 GB）——**直接下到租來的 GPU 機器上，不要放進這個資料夾**
2. 租一台 RTX 3090 等級的機器（純推論，約 NT$300–500）
3. `bash code/scripts/run_on_rented_gpu.sh /path/to/imagenet/val`

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
figures/    論文圖（PDF 給論文、PNG 給人看）
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

## 待自行確認的文獻

正式引用前要自己開 PDF 的（我這邊沒拿到全文）：

- **NORTH\*** (Maile et al., AutoML 2022, arXiv 2202.08539) — 最近鄰；是否在卷積濾波器上評估未確認
- **PCA-AE** (Pham et al., JMIV 2022) — 機制最像的先行工作；「逐階段凍結」的描述來自他人轉述
- **GOR** (Kurtz et al., BMVC 2024, arXiv 2306.10001) — 命名衝突；Eq. 2 與摘要措辭不符
