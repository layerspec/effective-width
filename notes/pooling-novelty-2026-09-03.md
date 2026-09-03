# 池化主張的新穎性查證（2026-09-03）

主張：全域平均池化後再估共變異數，在 108/108 層上一致地壓低量測到的維度；
這是水準差，不是形狀差。問題：有沒有人比較過「池化」與「逐位置取樣」？

**結論：最接近的是 Elmoznino & Bonner 自己的補充材料，但他們比的是另一個
對象、而且沒報水準。逐層配對比較仍然沒人做過。主張倖存，措辭要精確。**

## 各篇怎麼處理空間維度

| 文獻 | 樣本 × 特徵 | 有無比較 |
|---|---|---|
| Garg et al. 2019 | **N·H·W 個位置 × M 個通道**（全部位置，D/M ≥ 100，99.9% 變異）| 無；與我們的逐位置估計量同一對象（我們是隨機抽 16 個位置） |
| Elmoznino & Bonner 2024 | 主文：**全域平均池化**，N 張圖 × C；理由「we were primarily interested in the variance of image features … rather than the variance in those properties across space」 | **S6 Text**：把整張特徵圖攤平（N 張圖 × C·H·W）重做，結論「observed even when ED was computed on the entire flattened feature maps without pooling (though to a lesser extent)」；Fig S6.1、S6.6b。**沒有報兩種做法下 ED 的水準**，也沒有逐層配對 |
| Chen & Bonner 2025（Sci. Adv.）| 全域 **max** 池化 | 無 |
| Kong et al. 2022（V1 eigenspectrum）| 攤平 C·H·W，N 張圖為樣本 | 無 |
| Ghosh et al. 2022（power laws）| 未說明 | 無 |
| Recanatesi et al. 2019 | 未說明 | 無 |
| Bonner lab 2025（Nat. MI, cortex-aligned de novo）| 最後一層、池化後 | 無 |

三種對象要分清楚，論文裡已經是這樣寫（Eq. covpos / covpool），S6 的攤平是第三種：
1. 逐位置：樣本 = 位置，特徵 = 通道（Garg；我們）
2. 池化：樣本 = 圖，特徵 = 通道（E&B 主文；我們）
3. 攤平：樣本 = 圖，特徵 = C·H·W（E&B S6、Kong）——維度上界是 min(N, CHW)，
   跟通道數 C 沒有直接關係，不能除以 C

## 對論文的影響

- 主張 ② 的措辭：「池化與逐位置的**通道**共變異數，沒有人在同一批層上配對比較過」
  ——已改進 `paper/main.tex` 方法段，並在 `refs.bib` 的 E&B 條目註記 S6。
- 不能寫「E&B 沒有檢查過池化的影響」——他們檢查過（對他們的結論而言），
  只是沒報水準差。
- §V 那段「pooling … removes the terminal turnover … H2 is supported」仍是
  形狀主張，屬於規則 7 要重寫的部分，這次沒動。

## 侷限

同 r_max 那次：WebSearch／arXiv／PMC；沒查 Google Scholar 全清單。
剪枝文獻常用池化後的特徵做通道重要性（如 SE、HRank 的變體），但那不是
維度量測，未逐一檢查。
