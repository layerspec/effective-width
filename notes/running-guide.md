# 從零到有資料：租機器與取得 ImageNet

寫給第一次租雲端 GPU 的人。全程約 5 小時，其中你需要在場的大概 30 分鐘，
其餘是等它跑。花費約 **US$5–10**（含犯錯重來的餘裕）。

---

> **貼指令前先做這件事。** macOS 的 zsh 預設不把 `#` 當註解，整行貼上時
> 註解文字會被當成參數傳給指令（典型症狀：`Too many arguments`）。
> 執行一次 `echo 'setopt interactive_comments' >> ~/.zshrc`，開新視窗後即可。
> 本文件的指令區塊已經拿掉行內註解，說明都寫在區塊外。

## 0. 先理解你在租什麼

租來的是**一台別人機器上的 Linux 容器，附一張 GPU**。三件事要記住：

1. **從它存在的那一刻開始計費**，不管你有沒有在用。忘記關掉是最常見的損失。
2. **它是暫時的。** 你按下 destroy，上面所有東西都消失。
3. 所以流程永遠是：**把資料弄進去 → 跑 → 把結果弄出來 → 銷毀。**

## 平台選擇：這次用 RunPod，不要用 Vast.ai

我先前推薦 Vast.ai 是著眼於它的競價實例便宜。**對第一篇這種只跑幾小時的工作，
那個價差只有幾美元，不值得用介面複雜度去換。** RunPod 介面單純得多。

Vast.ai 的競價實例留給第二篇（1,200 GPU-hours，那時價差才有意義）。

---

## 1. 出門前（在自己電腦上做，免費）

### 1.1 Hugging Face 帳號與 ImageNet 授權

> **已完成（2026-09-03）。** 帳號 askia318，條款已接受，token 已建立並驗證。
> 驗證結果：repo 有 339 個檔案，驗證集是
> `data/validation-000NN-of-00014.parquet` 共 14 個 shard。
> 在 Mac 上重跑驗證：`source ~/hfenv/bin/activate` 之後執行下方指令。

1. 註冊 huggingface.co
2. 瀏覽器打開 `https://huggingface.co/datasets/ILSVRC/imagenet-1k`
3. 頁面上會要求同意 ImageNet 使用條款——**點下去就好，是即時的，不需審核**
4. 到 Settings → Access Tokens，建立一個 **read** 權限的 token，複製起來

那個 token 等一下要在租來的機器上用。**不要把它寫進任何會進 git 的檔案。**

### 1.2 SSH 金鑰

檢查你的 Mac 上有沒有：

```bash
ls ~/.ssh/id_ed25519.pub
```

沒有的話建一把：

```bash
ssh-keygen -t ed25519 -C runpod
```

三個提示都直接按 Enter（預設路徑、不設 passphrase）。然後印出公鑰：

```bash
cat ~/.ssh/id_ed25519.pub
```

輸出的那一整行 `ssh-ed25519 AAAA... runpod` 就是等下要貼到 RunPod 的內容。

### 1.3 把程式碼打包好

```bash
cd ~/Downloads/effective-width-claude
tar czf /tmp/layerspec.tar.gz --exclude=__pycache__ code/
```

---

## 2. 租機器

1. 註冊 runpod.io，儲值 **US$10**（預付制，用多少扣多少）
2. Settings → SSH Public Keys，把 1.2 那一行貼進去
3. Deploy 一台 Pod：

| 項目 | 選什麼 | 為什麼 |
|---|---|---|
| 雲別 | **Community Cloud** | 便宜約一半，這個工作不需要企業級可靠度 |
| GPU | **RTX 3090**（或 A5000、4090） | 純推論，24 GB 綽綽有餘；12 GB 其實就夠 |
| 數量 | 1 | |
| 模板 | 任何 **PyTorch** 官方模板 | 省去自己裝 CUDA |
| **Container Disk** | **80 GB** | ⚠️ 見下方 |
| Volume | 不需要 | 我們不需要跨 pod 保存 |

⚠️ **磁碟大小是最容易出錯的地方，而且開機後很難改。** 需要的空間：
parquet 快取 ~7 GB ＋ 解出的 JPEG ~7 GB ＋ 預訓練權重 ~2 GB ＋
CIFAR-10 ＋ 結果檔，再加上映像檔本身。**80 GB 是安全值，不要省。**

參考價位：Community Cloud 的 3090 大約 **US$0.22–0.30/hr**。

4. 按下 Deploy，等一兩分鐘機器起來
5. 面板上會給你一行 SSH 指令，長得像
   `ssh root@123.45.67.89 -p 40022 -i ~/.ssh/id_ed25519`
   **把主機位址和連接埠記下來**，等下 scp 要用

（RunPod 介面偶爾改版，細節可能和上面不完全一樣，但這幾個欄位一定找得到。）

---

## 3. 進去、把程式碼送上去

在你自己的 Mac 上：

```bash
scp -P 40022 /tmp/layerspec.tar.gz root@123.45.67.89:/workspace/
```

然後連進去：

```bash
ssh root@123.45.67.89 -p 40022
```

進去之後：

```bash
cd /workspace && tar --no-same-owner -xzf layerspec.tar.gz && cd code
pip install -q -r requirements.txt

nvidia-smi
PYTHONPATH=. python tests/test_core.py
```

**14/14 沒過就先停下來**，環境有問題，繼續跑只是浪費錢。

---

## 4. 抓資料

```bash
hf auth login
```

**先小試 2000 張**，確認整條路通：

```bash
python scripts/fetch_imagenet_val.py --out /workspace/imagenet_val --limit 2000
python -m layerspec.run --data /workspace/imagenet_val \
    --models resnet18 --limit 2000 --out /tmp/trial
```

跑得出東西（會警告取樣不足，正常，因為只有 2000 張）就繼續抓完整的：

```bash
python scripts/fetch_imagenet_val.py --out /workspace/imagenet_val
```

約 10–20 分鐘。結束時它會說寫了幾張，**應該是 50000**。

---

## 5. 正式跑

用 `nohup` 讓它在背景跑，這樣 SSH 斷線也不會中斷：

```bash
cd /workspace/code
nohup bash scripts/run_on_rented_gpu.sh /workspace/imagenet_val > run.log 2>&1 &
tail -f run.log
```

會依序跑：

| 階段 | 時間 |
|---|---|
| 正確性測試、煙霧測試 | 2 分鐘 |
| **Garg 重現關卡**（訓練 CIFAR VGG-16） | 約 1 小時 |
| 四個預訓練模型 | 約 1.5 小時 |
| **tier1 + tier2 共 16 個 timm checkpoint**（2026-09-03 新增） | 約 1 小時 |
| 兩個隨機初始化對照 | 約 45 分鐘 |
| 出圖、打包 | 2 分鐘 |

**Garg 關卡跑完會印出判定**（Pearson r 與 MAD）。r < 0.7 就該停下來找原因，
不要讓它繼續跑完——那代表管線或訓練有問題，後面的數字都不能用。
要跳過這關重跑其餘部分：`SKIP_GARG=1 bash scripts/...`

每個模型跑完會印 `r_max check: ... -> ok`；出現 **FAILED** 表示某種層的
r_max 算錯了，把 run.log 帶回來，不用停整趟。只跑前十個 checkpoint：
`TIERS=tier1 bash scripts/...`；不跑 checkpoint：`TIERS="" bash scripts/...`。

---

## 6. 把結果帶走，然後銷毀機器

腳本最後會印出一個 tarball 路徑。在你自己的 Mac 上：

```bash
scp -P 40022 root@123.45.67.89:/tmp/layerspec_results_*.tar.gz \
    ~/Downloads/effective-width-claude/results/
```

確認檔案真的在你電腦上、而且解得開：

```bash
cd ~/Downloads/effective-width-claude/results && tar tzf layerspec_results_*.tar.gz | head
```

**然後回到 RunPod 面板，把 Pod 按 Terminate/Destroy。**

⚠️ Stop 不等於 Destroy。停止的 pod 通常仍會收儲存費。要**銷毀**。

最後看一眼帳單，確認沒有東西還在跑。

---

## 常見錯誤

| 錯誤 | 後果 | 預防 |
|---|---|---|
| 忘記銷毀機器 | 持續扣款 | 拿到結果就立刻銷毀；設一個手機鬧鐘 |
| Container disk 開太小 | 抓資料抓到一半爆掉，很難補救 | 開 80 GB |
| 沒先用 `--limit 2000` 試 | 環境有問題時，浪費幾小時才發現 | 一定先小試 |
| 用 Stop 而不是 Destroy | 還在收儲存費 | 看清楚按鈕 |
| token 寫進檔案被 commit | 憑證外洩 | 只用 `hf auth login` |
| SSH 斷線導致工作中斷 | 前功盡棄 | 用 `nohup ... &` |

---

## 花費預估

| 項目 | 金額 |
|---|---|
| 實際運算約 5–6 小時 × US$0.25/hr | ~US$1.50 |
| 含犯錯、重跑、忘記關機的餘裕 | **編列 US$10** |

第二篇（正交再評估，約 1,200 GPU-hours）到時候再談，那時才輪到 Vast.ai
的競價實例，約 US$160。

## 第三輪（RunPod，2026-09-07 準備）

一趟跑完投稿門檻 A2、A5、A6、A9（`notes/submission-bar-2026-09-07.md`）。預先登記在
`analysis-plan.md` §9。在 pod 上：

```
git clone git@github.com:layerspec/effective-width.git && cd effective-width/code
pip install -r requirements.txt
python -c "import torchvision; torchvision.datasets.CIFAR10('../data', download=True); torchvision.datasets.CIFAR10('../data', train=False, download=True)"
# 只跑 A2、A5（軌跡 A9 在本機分多天跑，用 SKIP_TRAJ=1 略過；約 5–7 GPU-hours on 3090）：
SKIP_TRAJ=1 nohup bash scripts/run_round3_pod.sh > ../results/round3.log 2>&1 &
# 連 A6 一起（需要 ImageNet val 50k）：
hf auth login && python scripts/fetch_imagenet_val.py --out /data/imagenet_val
nohup bash scripts/run_round3_pod.sh /data/imagenet_val > ../results/round3.log 2>&1 &
```

中斷後重跑同一指令會略過已完成的 run（以 `epoch_100_layers.csv` 為準）。結束後：

```
tar czf round3_$(date +%Y%m%dT%H%M%SZ).tar.gz ../results/round3 ../results/fullval ../results/round3.log
```

拉回本機解到 `results/`，然後 `python3 scripts/checkpoint_analysis.py --results ../results`
（§9 會自動掃 `results/round3/*/trajectory_layers.csv`，A2／A5 的判定規則要另寫 §12，見 plan §9）。
私人 repo 的 clone 需要把 pod 的 SSH 公鑰加到 layerspec 帳號，或用 https + token。

### 2026-09-08 實跑筆記（3090，CPU 少）

- 三個種子平行跑沒有加速（GPU 2%，瓶頸是 CPU 發指令），每 epoch 約 51 s；`--gpu-data`
  在這種 pod 上也救不了。24 個 run 約 11–12 小時、US$4。下次選 vCPU 多的機型再考慮平行。
- 私人 repo 用 deploy key（勾 write）讓 pod 自己推結果；`*.pt` 在 .gitignore，所以分解量測
  （`decompose_round3.sh cuda`）要在 pod 上做完再推 CSV。
- 跑完自動推送＋停機的守護程序（貼在 pod 上，長跑啟動後）：

```
cd /workspace/effective-width && git config user.name layerspec && git config user.email 325815311+layerspec@users.noreply.github.com
nohup bash -c 'until grep -q "round 3 done" results/round3.log; do sleep 300; done; cd code && PY=python bash scripts/decompose_round3.sh cuda > ../results/round3_decompose.log 2>&1; cd .. && git add results/round3 results/round3_decompose results/round3_decompose.log && git commit -q -m "Round 3 results from the pod" && git fetch -q origin && git rebase origin/master && git push origin master; runpodctl stop pod $RUNPOD_POD_ID' > results/autostop.log 2>&1 &
```

- **push 前一定要 `git fetch && git rebase origin/master`**（2026-09-08 的教訓：pod clone 後 Mac 又推了 39 個 commit，
  pod 跑完的 `git push` 被 non-fast-forward 拒絕，結果卡在 pod 的 /workspace 裡；守護程序照樣停機，外面看不出來）。
  補救：從 Mac `git fetch ssh://root@<ip>:<port>/workspace/effective-width master` 再 cherry-pick，不必在 pod 上補 key。
- **Stop 再 Start 會重置容器磁碟（/root）**：deploy key、`/root/.runpod/restkey`、known_hosts 全部消失，只有 /workspace 留著。
  不想重貼 key 的話，結果一律從 Mac 端拉。
- **`runpodctl stop pod` 在 2026-09-08 的 pod 上不能用**（config 後仍 Unauthorized）；改用 REST API：
  先 `echo '<API key>' > /root/.runpod/restkey && chmod 600 /root/.runpod/restkey`（key 只貼在 pod 裡，
  **不要貼進對話**），守護程序結尾用
  `curl -s -X POST https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop -H "Authorization: Bearer $(cat /root/.runpod/restkey)"`。
  先用 GET 同一網址確認回 200。key 權限 Read & Write 即可。
- Stop 之後磁碟費約 US$0.3/天；拿到結果後到網頁 Terminate，並 Revoke 用過的 API key。

## A18（用尺定寬度再重訓；plan §9.5／§9.6）—— pod 一趟，2026-09-09 準備

寬度已算好並入庫：`results/a18/widths.json`（seed 0 全寬權重、6,400 張**訓練**影像、16 個位置；
ruler 5.09M 參數 = 全寬的 34.5%，uniform 4.99M（因子 0.582），ruler95 0.63M）。在 pod 上：

```
git clone git@github.com:layerspec/effective-width.git && cd effective-width/code
pip install -r requirements.txt
python -c "import torchvision; torchvision.datasets.CIFAR10('../data', download=True); torchvision.datasets.CIFAR10('../data', train=False, download=True)"
nohup bash scripts/run_a18_pod.sh > ../results/a18.log 2>&1 &          # 9 run，3090 約 9 小時
# 跑完後（log 出現 "A18 done"）：每個 checkpoint 的分解量測（含 rho_align），約 2–3 小時
PY=python bash scripts/decompose_a18.sh cuda > ../results/a18_decompose.log 2>&1
```

守護程序（跑完自動分解、推送、停機）—— **push 前一定 fetch＋rebase**：

```
cd /workspace/effective-width && git config user.name layerspec && git config user.email 325815311+layerspec@users.noreply.github.com
nohup bash -c 'until grep -q "A18 done" results/a18.log; do sleep 300; done; cd code && PY=python bash scripts/decompose_a18.sh cuda > ../results/a18_decompose.log 2>&1; cd .. && git add results/a18 results/a18.log results/a18_decompose.log && git commit -q -m "A18 results from the pod" && git fetch -q origin && git rebase origin/master && git push origin master; curl -s -X POST https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop -H "Authorization: Bearer $(cat /root/.runpod/restkey)"' > results/a18_autostop.log 2>&1 &
```

`*.pt` 在 .gitignore：17 個 epoch × 9 run 的 checkpoint（每個 20–60 MB）留在 pod 的 /workspace，
分解 CSV 才推。想留權重就在 Terminate 前 `rsync` 回 `results/a18_pt/`。
分析：`checkpoint_analysis.py` §16（P9.6–P9.9）與 §17（P9.10–P9.13）待資料到齊後寫，判定規則照 plan。
