# `layerspec.realloc`：可移植的寬度調撥套件設計（2026-09-16，作者要求「小型通用版」）

作者的標準：理論要有嚴格證明；實務要有套在真實架構上的實作並推上 GitHub；一兩種常用架構即可；
理論要能被別人移植到其他架構。所以控制器不能寫死在 VGG 或 ResNet 裡，要拆成「與架構無關的運算子」＋「架構配接器」。

## 核心抽象：Site（一個可調的寬度）

```
Site(
  name,                # 例 "layer1.0.conv1"
  producer,            # 產生這個寬度的模組（Conv2d/Linear 的輸出通道）
  norm,                # 緊接的 BatchNorm/LayerNorm（可 None）
  activation_hook,     # 讀「後激活」張量的位置（ReLU/GELU 輸出；ResNet 共用 ReLU 時用 BN 輸出＋自算激活）
  consumers,           # 消費這個寬度的模組清單：(module, 沿哪個軸, 之後接哪個 norm 以摺常數)
  tie_group,           # 綁在一起的寬度（殘差流：同 stage 的 block 輸出；第一版只調 tie_group=None 的 site）
  cost,                # 改一個通道的參數／FLOPs 成本（由 producer、consumers 的形狀與 H·W 算出）
)
```

任何架構只要能列出它的 Site 清單，四個運算子就能用：

| 運算子 | 輸入 | 做什麼 | 對應命題 |
|---|---|---|---|
| `measure(model, sites, loader)` | 幾次前向 | 每個 site 的後激活均值與共變異數 Σ_a、下一層的 Σ_x（未滿足變異） | — |
| `shrink(model, site, m, stats)` | m | CSSP 選 S、把移除通道的線性重建摺進 consumers（常數摺進其 norm） | A |
| `grow(model, site, k, stats)` | k | 新通道沿 Σ_x 在列空間正交補的頂特徵向量初始化、consumers 對應行歸零、norm 新通道 γ=1 β=0 | B |
| `allocate(sites, stats, budget, cost)` | 預算 | 反向注水（連續解＋離散貪婪交換），逐層錯開、鏈式、速率上限 | C |
| `Controller(model, sites, budget, triggers)` | 訓練迴圈 | `step(epoch, calib_loader)`：依觸發讀統計、注水、施作 | 時程 |

## 配接器（論文只做這三個，其餘由使用者照樣寫 30 行）

| 配接器 | Site 從哪裡來 | 消費者 | 常數摺到 | 已驗證 |
|---|---|---|---|---|
| `sequential_conv`（VGG 類） | 每個 Conv→BN→ReLU | 下一個 Conv（k×k）、末層 Linear | 下一個 Conv 的 bias | P1（`fold_cssp.py`） |
| `resnet_internal`（BasicBlock、Bottleneck） | block 內 conv1（與 conv2）輸出 | block 內下一個 conv | 下一個 BN 的 running_mean | `fold_resnet.py` |
| `mlp`（ViT／DeiT／ConvNeXt） | fc1 輸出 | fc2 | fc2 的 bias | `fold_mlp.py` |

`discover_sites(model)` 自動找這三種樣式；找不到的架構回傳空清單並提示寫配接器。

## 保證與測試（每條命題一個測試）

- `test_grow_preserves_function`：放後輸出逐位相等。
- `test_shrink_dead_channel_exact`：砍常數通道後輸出相等。
- `test_fold_equals_hook`：摺入的窄網路與 hook 重建的 logit 差 < 1e-3（無池化）；有 max-pool 時報告差值。
- `test_cssp_bounds`：tr(R_S) 落在 [Σ_{i>m}λ_i, (m+1)Σ_{i>m}λ_i]。
- `test_waterfilling_kkt`：連續解滿足邊際相等；離散貪婪不增 Φ。
- `test_linearisation`：砍一般通道的 logit 變化與命題 A 線性化 ρ > 0.95。

## 對外形式

`pip install layerspec` 已存在（profile／decompose）。加 `layerspec.realloc`，README 一頁：三行用法
（`sites = discover_sites(model)`；`ctrl = Controller(model, sites, budget="params:0.67")`；訓練迴圈裡 `ctrl.step(epoch, calib)`）、
一段「移植到新架構：實作 Site 清單」、三個配接器當範例。Release tag 與論文同時。

## 現有程式怎麼併

`cssp_check.py`／`fold_cssp.py`／`fold_mlp.py`／`fold_resnet.py` 是三個配接器的原型，共用的 `greedy_cssp`、`Stats`、
`fold` 邏輯抽到 `layerspec/realloc/`，腳本改為呼叫套件。改 `layerspec/` 前照規則 `pgrep -f trajectory_cifar`。
