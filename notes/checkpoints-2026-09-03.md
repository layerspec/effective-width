# Checkpoint 清單（2026-09-03）

目的：解決 n = 1（`review-2026-09-03.md` §4），並讓 r_max 的修正在
r_max/C ≠ 0.25 的架構上被測到。全部 ImageNet-1k 分類權重，來源 timm（HF
`timm/<name>`，名稱已在 2026-09-03 用 HF API 核對存在）。

成本：每個模型一趟 ImageNet val 前向，3090 上 ResNet-50 等級約 2–3 分鐘；
20 個模型 ≈ 1 GPU-hour，加資料搬運，**< US$3**。

## 分層優先序

### Tier 1 — 必跑（10）

**A. 同架構、不同訓練配方（n=1 的直接解法）** —— ResNet-50 × 6，r_max 全同，
差異只來自訓練：

| timm 名稱 | 配方 |
|---|---|
| `resnet50.tv_in1k` | torchvision V1（已有，作對照基準） |
| `resnet50.tv2_in1k` | torchvision V2 配方 |
| `resnet50.a1_in1k` | Wightman RSB A1（600 ep, BCE） |
| `resnet50.a3_in1k` | RSB A3（100 ep, 160px，較弱） |
| `resnet50.gluon_in1k` | GluonCV 配方 |
| `resnet50.fb_ssl_yfcc100m_ft_in1k` | 半監督預訓練後微調（不同資料） |

**B. r_max/C 不同的 bottleneck 架構** —— 若 k\*/r_max 維持水準而 k\*/C 跟著
r_max/C 走，修正就是對的：

| timm 名稱 | 受限層 r_max/C | 說明 |
|---|---|---|
| `resnet101.tv_in1k` | 0.25 × 33 層 | 更多受限層 |
| `wide_resnet50_2.tv_in1k` | **0.5** | bottleneck 寬度 2×，擴張只 2× |
| `resnext50_32x4d.tv_in1k` | **0.5** | 同上；grouped 3×3 不受限（每組 4·9 > 4） |
| `mobilenetv2_100.ra_in1k` | **1/6** × 16 層 | inverted bottleneck：第一個 1×1 擴張 6×；投影 1×1 與 depthwise 都不受限 |

### Tier 2 — 有預算就跑（6）

| timm 名稱 | 為什麼 |
|---|---|
| `efficientnet_b0.ra_in1k` | 1/6 擴張 + SE + SiLU |
| `mobilenetv3_large_100.ra_in1k` | 擴張比 3–6 混合、Hardswish、SE |
| `vgg16.tv_in1k` | 無 BN 的 VGG-16，隔離 BN 的影響 |
| `resnet34.a1_in1k` | basic block，幾乎無受限層，另一個「無效應」對照 |
| `convnext_tiny.fb_in22k_ft_in1k` | 同架構、不同預訓練資料（對 convnext_tiny.fb_in1k） |
| `densenet121.tv_in1k` | registry 已有；dense block 首層 1×1 是 0.5 受限 |

### Tier 3 — 補齊分佈用（4）

`resnet50.a2_in1k`、`resnet50.c1_in1k`、`resnet18.a1_in1k`、`vgg19_bn`
（torchvision）、`convnext_small.fb_in1k`、`regnety_016.tv2_in1k`（grouped 3×3
+ SE，無受限層）。

## 需要的程式改動（小）

1. `models.py`：加 timm 路徑，名稱形如 `timm:resnet50.a1_in1k`，
   `timm.create_model(name, pretrained=True)`；`requirements.txt` 加 `timm`。
2. `hooks.py`：SE 模組裡的 1×1 conv 作用在全域池化後的 1×1 空間，n 只有影像數，
   會被 n/C 門檻擋掉，但最好在 attach 時就標記 `kind="se"` 排除，避免混進
   conv 計數。
3. `hooks.py`：另外 hook **block 輸出**（timm 的 `Bottleneck`／`InvertedResidual`
   ／ConvNeXt `Block` 的 forward 輸出，殘差相加之後），給「conv 輸出 vs block
   輸出不可比」那一節用。這是量測點的差異，不是新統計量。
4. 每個模型跑完立刻檢查 `max k*(0.999)/r_max ≤ 1`，超過就是 r_max 對該架構
   算錯（timm 的某些層是 `Conv2dSame`，是 `nn.Conv2d` 子類，應該沒問題，但要看）。

## 不需要的

- ShuffleNet-v1 之類的 grouped 1×1 擴張：g·min((C_in/g)k², C_out/g) =
  min(C_in k², C_out)，g 項在代數上消掉，沒有東西可驗證（見
  `rmax-novelty-2026-09-03.md` §4）。
- CLIP 權重的 ResNet-50（`resnet50_clip.*`）：分類頭與訓練目標不同，混進來
  會汙染「同架構不同配方」那一組。
