# Elmoznino & Bonner 逐層 ED 作為第二道驗證關卡（2026-09-08）

問題：E&B 2024（PLOS Comput Biol 20(1) e1011792）有沒有公開 ImageNet 訓練
ResNet-50（或任何 torchvision 模型）的逐層 effective dimensionality
（participation ratio），可以拿來對照我們「hook block 輸出 → 全域平均池化 →
PR」的管線？

**結論：有，而且是完整的逐層 CSV，不是只有圖。已下載到 `refs/eb2024/`。
用同一組 torchvision 權重（IMAGENET1K_V1）、同一個量測位置（block 輸出）、
同一個估計量（avg-pool 後 PR），我們 16 個 ResNet-50 block 與他們的
log–log r = 0.999、中位 |Δlog| = 0.063、我們一致高 6.5%。管線通過。
Chen & Bonner 2025 沒有釋出任何逐層 ED，只能借它的層命名慣例。**

## 來源

| 項目 | URL |
|---|---|
| 論文 | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011792 |
| Data Availability（原文） | "All code used for this project has been made publicly available on GitHub: https://github.com/EricElmoznino/encoder_dimensionality"；神經資料經 BrainScore 0.2 自動下載，fMRI 在 https://osf.io/ug5zd/ |
| 程式碼＋結果 | https://github.com/EricElmoznino/encoder_dimensionality ，commit `ddf33ad`（2023-11-24 "Updated figures for copyright issues"），clone 下來 656 MB，`results/` 裡的 CSV 全部在 git 內 |
| 補充材料 | S1–S10 Text（PLOS 頁面列出）；**沒有逐層 ED 的補充表格**，數字只在 `results/*.csv` |
| Chen & Bonner 2025 | Sci. Adv. 11:eadw7697，https://doi.org/10.1126/sciadv.adw7697 （science.org 回 403，改讀 PMC12219468）；程式碼 https://github.com/zche377/universal_dimensions ，Zenodo v1.0.2 https://doi.org/10.5281/zenodo.15390375 ；demo 快取 OSF `wvj3s` |

## 他們到底怎麼算（讀程式碼，不是讀論文）

論文正文只說「10,000 images from the ImageNet validation set」與「global
average pooling」，其餘細節全在程式碼：

| 環節 | 實作（`refs/eb2024/eb_*.py` 有原檔） | 我們 |
|---|---|---|
| 影像 | `utils.get_imagenet_val(num_classes=1000, num_per_class=10)`：從 brain-score 的 `imagenet2012-val.hdf5` 取 index `50·i + {0..9}`，即**每類前 10 張**（hdf5 內部順序，見「開不了的東西」）；存成 PNG | ImageNet val 全 50,000 張 JPEG |
| 前處理 | model-tools `load_preprocess_images`：`Resize((224, 224))` **直接拉成正方形，不保持長寬比、不 center-crop**，ToTensor，ImageNet mean/std；Taskonomy 用 256 | `Resize(256)` → `CenterCrop(224)` → 同 mean/std |
| 權重 | `torchvision.models.resnet50(pretrained=True)`，Python 3.7 時代；`pretrained=True` 在 torchvision 的 `handle_legacy_interface` 固定對到 `IMAGENET1K_V1`（已核對 v0.13 原始碼） | `ResNet50_Weights.IMAGENET1K_V1` —— **同一組權重** |
| 量測位置 | 層名 `layer{1..4}.{i}.relu`（ResNet-50 共 16 個）。這是 block 內**共用的** `nn.ReLU`；model-tools 的 forward hook 是 `target_dict[name] = output`，同一次前向被呼叫三次就**覆寫三次，留下最後一次** = 殘差相加後的 ReLU = block 輸出 | `kind="block"` 的 `Bottleneck` 輸出 —— 同一個張量 |
| 池化 | `F.adaptive_avg_pool2d(x, 1)`，N 張圖 × C | `out.mean(dim=(2,3))`，同 |
| 譜 | `sklearn.decomposition.PCA(random_state=0).fit(X).explained_variance_`（無偏 /(n−1)），全秩 | 累積共變異數再 eigh；PR 對常數倍不敏感 |
| ED | `eigspec.sum()**2 / (eigspec**2).sum()` | `metrics.participation_ratio` 同式 |
| 其他欄位 | `80% variance`（達 80% 變異的 PC 數）、`alpha`（50 個對數等距 rank 的 log–log 斜率；`utils.fix_alpha` 另有去尾版本，不確定 CSV 用的是哪個） | k*(0.9/0.95/0.99/0.999)、`powerlaw_alpha`（不同擬合區間，**不可直接比**） |

其他模型的層：ResNet-18 `layer{1..4}.{0,1}.relu`（BasicBlock 輸出，同理最後
一次呼叫）；VGG-16 `features.{18,20,22,25,27,29}`（**ReLU 模組**，post-
activation，不是 conv）；AlexNet `features.{1,4,7,9,11}`（ReLU）；SqueezeNet
`features.{4,6,8,10,12}`（Fire 模組）。

## 下載的檔案（`refs/eb2024/`，3.4 MB）

原始檔名含 `|` 與 `:`，改名時保留語意。**注意命名歷史**：舊版程式的 `pooling:True/False` 是「max 池化／攤平」，新版的 `pooling:avg/none` 是「avg 池化／隨機投影到 1024 維」。手稿 notebook（`figures/manuscript/results.ipynb`）主圖讀 `pooling:avg`，S6 攤平分析讀 `pooling:False`。

| 檔案 | 原檔 | 內容 |
|---|---|---|
| `eigmetrics_imagenet_pooling-avg.csv` | `eigmetrics\|dataset:imagenet\|pooling:avg\|grayscale:False.csv` | **主文用的那份**。536 列 = 40 組（架構,任務,種類,來源）× 層；欄位 `80% variance, alpha, effective dimensionality, layer, architecture, task, kind, source`。含 ResNet-50/18 torchvision 有訓練＋隨機、Barlow Twins、11 個 VVS ResNet-18、24 個 Taskonomy ResNet-50 |
| `eigmetrics_imagenet_pooling-avg_additional.csv` | 同上 `\|additional:True` | AlexNet、VGG-16、SqueezeNet 各有訓練／隨機，32 列 |
| `eigmetrics_imagenet_pooling-max.csv` | `pooling:True` | 同 536 列，**全域 max 池化**——Chen & Bonner 2025 用的估計量，日後要對 C&B 可用這份 |
| `eigmetrics_imagenet_pooling-none-flattened.csv` | `pooling:False` | 攤平 C·H·W（S6 Text 那個對象），ED 可到 600+，分母是 min(N, CHW) 不是 C，**不能除以 C** |
| `eigmetrics_imagenet_pooling-randproj1024_additional.csv` | `pooling:none\|additional:True` | 隨機投影 1024 維，只有 additional 模型 |
| `eigspectra_imagenet_pooling-avg_torchvision.csv` | 從 34 MB 的 `eigspectra\|…pooling:avg…` 抽 `source=PyTorch` 的列 | 完整特徵譜，欄位 `n, variance, layer, …`；ResNet-50 每層長度 = C（256/512/1024/2048），可自己重算任何 k* |
| `eigspectra_imagenet_pooling-avg_additional.csv` | 同上 additional 版 | AlexNet/VGG/SqueezeNet 完整譜 |
| `manuscript_fig_data.csv` | `figures/manuscript/data/data.csv` | 主圖資料：ED＋SNR＋MRR＋encoding score，ED 值與 pooling-avg 檔完全相同 |
| `eb_generators.py`, `eb_eigenspectrum.py`, `eb_utils.py` | `activation_models/generators.py` 等 | 上表出處，留底 |

沒抓：grayscale 變體、`dataset:majajhong2015` 變體、encoding／RSA／n-shot／
manifold 結果、ZCA 反例。都在 repo，要時再 clone。

## 對照：ResNet-50 IMAGENET1K_V1，16 個 block 輸出，avg 池化 PR

我們 = `results/resnet50_layers.csv` 的 `pooled_participation_ratio`（50,000 張）；
E&B = `eigmetrics_imagenet_pooling-avg.csv` 的 `effective dimensionality`（10,000 張）。

| block | C | 我們 | E&B | 我們/E&B | n/C 我們 | n/C E&B | 隨機：我們 | 隨機：E&B |
|---|---|---|---|---|---|---|---|---|
| layer1.0 | 256 | 8.31 | 8.05 | 1.03 | 195 | 39 | 2.63 | 2.12 |
| layer1.1 | 256 | 9.69 | 9.36 | 1.04 | 195 | 39 | 2.53 | 2.20 |
| layer1.2 | 256 | 10.30 | 9.95 | 1.04 | 195 | 39 | 1.75 | 1.79 |
| layer2.0 | 512 | 8.09 | 8.31 | 0.97 | 98 | 20 | 1.60 | 1.68 |
| layer2.1 | 512 | 14.49 | 13.14 | 1.10 | 98 | 20 | 1.45 | 1.55 |
| layer2.2 | 512 | 16.76 | 15.34 | 1.09 | 98 | 20 | 1.28 | 1.41 |
| layer2.3 | 512 | 19.24 | 17.68 | 1.09 | 98 | 20 | 1.22 | 1.27 |
| layer3.0 | 1024 | 20.07 | 18.09 | 1.11 | 49 | 10 | 1.17 | 1.23 |
| layer3.1 | 1024 | 29.09 | 27.16 | 1.07 | 49 | 10 | 1.13 | 1.19 |
| layer3.2 | 1024 | 31.89 | 30.50 | 1.05 | 49 | 10 | 1.11 | 1.13 |
| layer3.3 | 1024 | 35.37 | 33.84 | 1.05 | 49 | 10 | 1.09 | 1.10 |
| layer3.4 | 1024 | 40.30 | 38.04 | 1.06 | 49 | 10 | 1.08 | 1.08 |
| layer3.5 | 1024 | 57.70 | 52.13 | 1.11 | 49 | 10 | 1.06 | 1.07 |
| layer4.0 | 2048 | 61.78 | 57.25 | 1.08 | 24 | 5 | 1.07 | 1.07 |
| layer4.1 | 2048 | 78.87 | 70.48 | 1.12 | 24 | 5 | 1.06 | 1.06 |
| layer4.2 | 2048 | 123.14 | 117.46 | 1.05 | 24 | 5 | 1.06 | 1.06 |

- n = 16：log–log Pearson r = 0.999，Spearman ρ = 0.997，中位 |Δlog| = 0.063，
  最大 0.112（layer4.1）。依規則 9：我們自己這條曲線的相鄰層階差中位數是
  0.145，對照誤差不到它的一半。
- **我們一致高 6.5%（中位比值 1.065），15/16 層同號。** 可能的來源沒有分開：
  (a) 50k vs 10k 張——有限樣本讓 PR 偏低，樣本多者偏高；(b) center-crop vs
  直接拉伸；(c) JPEG vs hdf5→PNG。要分開得用他們那 10k 子集重跑（便宜，但
  見下）。目前只能寫「系統性 +6%，方向與樣本數效應一致，未分離」。
- 隨機初始化那欄：他們是**一個** init，我們是固定種子的一個；量級一致
  （1.1–2.6），不能再多說。
- **規則 2 的門檻要誠實講。** 我們的池化估計量 n/C ≥ 50 只在 layer1–2 成立
  （7 個 block），layer3+ 是 `pooled_n_over_C_ok=False`。E&B 自己 layer4 只有
  n/C ≈ 5。所以這個關卡的正式版只能寫 7 個 block（max |Δlog| = 0.098）；
  16 個 block 的 r = 0.999 是「兩條同樣取樣不足的曲線互相吻合」，可以放在
  探索性段落，不能拿來說 layer4 的水準是對的。

## 怎麼寫進論文

- 這是**管線驗證**（hook 位置、池化、估計量三者一起對），不是任何承重主張
  的證據；與 Garg CIFAR-10 關卡並列，措辭是「reproduces E&B's published
  per-block values to within 0.11 in log ED」。
- 「重現」二字：規則 11 只准用在 Garg 關卡。這裡是同權重、同資料集、不同
  影像子集、不同前處理，寫 "agrees with" 不寫 "reproduces"。
- 層名對齊只要 `layer.replace('.relu','')` → 我們 `kind=='block'` 的 `layer`
  欄；已在上表做過。VGG-16／AlexNet 要對的話，E&B 的層是 **ReLU 模組**，得
  對我們 `kind=='act'` 的列（`features.N::act`），不是 conv 列——尚未做。
- E&B 的 max 池化檔（`pooling-max.csv`）是免費的第三個估計量對照；我們目前
  沒算 max 池化，加一個 accumulator 就有。

## Chen & Bonner 2025

- 沒有逐層 ED。方法：NSD（COCO）刺激，PCA 在 72,128 張 unshared 影像上做、
  保留到 `torch.linalg.matrix_rank` 的秩（`src/lib/computation/_pca.py`），
  全域 **max** 池化，模型來自 torchvision 0.18 ＋ timm 1.0.3 ＋ VISSL；補充
  材料只有 Figs S1–S13、Tables S1–S3，沒有特徵值表。
- 有用的一樣東西：`src/lib/models/model_layers.csv`（523 列）。ResNet-50 的
  層寫成 `layer1.0.relu_2`——bonner-libraries 對重用的 ReLU 加**呼叫序號**
  後綴，跟我們 `call_index` 修的是同一個坑，可引用來佐證這個問題是真的。
- OSF `wvj3s` 只看到兩層：`bonner-caching/{data, stimulus_sets, miscellaneous}`、
  `bonner-models/{models, features}`。沒往下開；就算有快取特徵也是 NSD 的，
  對 ImageNet ED 關卡無用。

## 開不了／沒查的東西

- science.org 直接 403；C&B 的方法段是從 PMC 讀的，Data availability 原文
  已核對。
- PLOS 的 S1–S10 Text 沒有逐一開，只憑頁面清單與 repo 的 `saved/appendix_*`
  檔名確認沒有逐層 ED 表；`appendix_ED-vs-encoding-bestlayer-{pooled,nopooling}`
  是 S6 那組圖。
- brain-score 的 `imagenet2012-val.hdf5` 在他們的 S3，需要憑證；**類內前 10
  張的順序無法確認**（是檔名排序還是別的），所以「用同一 10k 子集重跑」
  現階段做不到精確版，只能做「每類前 10 張（檔名序）」的近似版。
- 我第一次猜的 PMC id 是錯的（打到別篇），E&B 的細節全部改從 PLOS 頁面與
  程式碼取得；正文的方法段沒有寫前處理，上表的前處理是**從程式碼讀出來的**。
- 沒查 Google Scholar 引用清單看有沒有第三方重算過 E&B 的數字。
