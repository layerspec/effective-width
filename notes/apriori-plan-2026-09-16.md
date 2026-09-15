# 先驗剖面：把第一篇從「量測報告」變成「有理論主張」（2026-09-16）

**決定**（作者，09-16）：三份敵意審稿（`notes/hostile-review-2026-09-15.md`、`review/*.docx`）一致認為現稿是
「嚴謹的量測報告、沒有新理論或新方法」，TPAMI 機率估 45–50%（major revision 以上）。作者選路 2：不改投
TNNLS，維持 TPAMI 當代表作，把「未訓練剖面可預測」做成理論主張併入第一篇。RunPod 餘額 US$60.37。

## 主張（要驗證的）

> 在初始化時，一層的有效寬度剖面由架構（`r_max`、kernel 大小、stride、殘差、Prop. S2）與影像的二階統計
> 決定，可以在訓練前算出；plain／basic／dense 家族訓練後保留這個順序、只加水準；bottleneck 在 ImageNet
> 配方下重排（原因未定）。

已有的證據（現稿）：初始化核近等向（erank 0.86）、初始寬度跟隨資料因子（ρ 0.94）、隨機初始化的 Σ_patch
隨深度失秩（每個家族的隨機剖面都下降）；ResNet-50 隨機初始化在 1/f 雜訊上保留 ImageNet 順序 ρ 0.88
（R–R 天花板 0.94），**但水準不同**（0.32 vs 0.23）——二階統計只解釋順序，水準要另外交代。

## 可行性檢查（先做，全部在 Mac 上，不花錢）

1. **單層 ReLU 遞推**：給定量到的 conv 輸出共變異數（需要矩陣，不是只有譜；查 `results/*.npz` 是否存了
   矩陣，沒有就在 6,400 子集上重量幾層），用高斯假設下的 arc-cosine 公式
   Cov(relu z) 預測後激活共變異數的 k*，對照量到的後激活 k*。這一步驗證「高斯＋ReLU」在有限寬度下
   準不準；預期第一層（RGB patch）最差。
2. **patch 組裝**：Σ_patch^(ℓ+1) 需要層 ℓ 後激活的**跨位置**共變異數（k_h·k_w 個位移的自相關）。管線現
   在只抽樣位置、不存位移共變異數；要加一個 hook 存 (C·k·k)×(C·k·k) 的 Σ_patch（decompose 已經在
   存了——`results/decompose/*_decompose.csv` 只有摘要，看 `scripts/decompose_check.py` 有沒有留矩陣）。
3. **整網遞推**：從影像的空間共變異數（1/f，或直接用 6,400 張的 27×27 RGB patch 共變異數）出發，逐層
   做「隨機 W 投影（期望值或幾個 W 樣本）→ ReLU 遞推 → 位移組裝 → 池化／stride／殘差相加」，輸出
   每層預測的 k*(0.95)/r_max，對照 11 個家族量到的隨機剖面（`results/local6400/random*/`），指標是
   每家族的 Spearman 對 R–R 天花板（0.79–0.98）。
4. 判定：預測 ρ 到達 R–R 天花板的八成以上（大多數家族 ≥ 0.75）→ 理論成立，進第一篇；否則回到路 1。

## 文獻（動手前要查，避免重複 r_max 的教訓）

- Poole 2016、Schoenholz 2017（mean-field 訊號傳播）、**Xiao et al. 2018（CNN 的 mean-field 與 dynamical
  isometry）**——共變異數逐層遞推的機制已存在；我們的新東西是「用它預測有效寬度剖面，並對照訓練後
  的順序」，不是遞推本身。論文要這樣寫。
- Novak 2019、Garriga-Alonso 2019（conv NNGP）：無限寬極限下通道 iid，通道共變異數趨於對角——與量到的
  初始 0.28·r_max 矛盾，所以要的是**有限寬度**的秩傳播（Prop. S2 ＋ 隨機投影），不是 NNGP 極限。
- Daneshmand 2020／2021（BN 讓初始秩 ≥ √width）、Feng 2022（Jacobian 秩隨深度單調降）、Saxe 2014
  （深線性：訓練沿輸入主方向）、Xiao 2020（CK 譜隨深度）——「初始秩隨深度降」是已知現象，**不得寫成
  發現**；可寫的是「逐層數值預測對上 11 個家族的量測」與「訓練後保留順序」。
- 查：有沒有人已經算過「conv 網路初始化時逐層有效秩／PR 的剖面」並對照過訓練後的（關鍵字：effective
  rank at initialization depth profile; conjugate kernel spectrum depth; rank propagation convolutional）。

## 之後的花費

- A17 縮減版（ImageNet-1k basic-[6,6,6,6] vs ResNet-50 同配方）：2 種子各 US$120–150；餘額 60.37，
  一種子各一或縮短排程（40 epoch）約 US$50–70 可以先做「有無重排」的定性判定。留到可行性檢查通過再決定。

## 不做的

- 不改投 TNNLS／TMLR（除非第 4 步失敗）。標題暫不動。D1 影像 bootstrap 仍要做（便宜），與這條線無關。
