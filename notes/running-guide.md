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

同一趟 pod 順便跑 **P9.10 的 44 個分解**（22 組 ImageNet 權重 × 訓練／隨機，含 rho_align；3090 約 7 小時）：
```
python scripts/fetch_imagenet_val.py --out /data/imagenet_val_6400 --limit 6400   # 或從 Mac rsync data/imagenet_val_6400（647 MB）
PY=python bash scripts/decompose_align_pod.sh /data/imagenet_val_6400 cuda > ../results/decompose_align.log 2>&1
```
守護程序的 `git add` 加上 `results/decompose_align`。本機另有 `results/decompose_align_local.sh`
（等 cifar 隊列跑完後重做 24 個 round-3 最終網路的分解，含 rho_align，給 P9.12）：
`nohup caffeinate -i results/decompose_align_local.sh >> results/decompose_align_local.log 2>&1 &`

### 2026-09-10 實跑筆記（4090、48 vCPU、Community Cloud）

- **CPU 執行緒一定要上限。** 3 個訓練程序各開 24 條 torch 執行緒＋1 個分解程序 76 條，48 核 load 85，
  10 分鐘沒印出一個 epoch；`OMP_NUM_THREADS=4` 後每 epoch 3–15 秒。分解程序沒上限時 ResNet-18 的
  特徵分解跑 35 分鐘（Mac 上 3 分鐘）；`OMP_NUM_THREADS=16` 後正常。三個腳本（`run_a18_pod.sh`、
  `decompose_a18.sh`、`decompose_align_pod.sh`）現在都內建上限。
- **多 vCPU 的機型才值得並行**：`PAR=3` 三種子並行，9 個 VGG run 1.5 小時（上次 CPU 少的 3090 要 12 小時）。
- **從 Mac 用 ssh 遠端 `pkill -f <字串>` 會砍掉自己的 ssh 連線**（遠端指令列含同一字串）。
  用 `ps -eo pid,cmd | grep "[a]utostop" | awk` 取 PID 再 `kill`，且啟動指令（含腳本名的字面）
  跟 kill 分開兩次 ssh。同一原因下守護程序曾被誤開 3 個又誤砍成 0 個；每次都用 `ps` 列全指令列確認。
- 在 ssh 一行指令裡 `nohup setsid ... &` 之後再接 `sleep`，ssh 常掛住不回；改成把啟動流程寫成檔案
  `scp` 上去再 `ssh 'bash file'`，穩定。
- 三個程序同時 `torchvision.datasets.CIFAR100(download=True)` 沒有互踩（8 分鐘下載完），但 pod 對
  toronto.edu 的下載只有 350 kB/s；ImageNet 子集改由 Mac `rsync` 上去（647 MB 約 15 分鐘；反向拉
  2.2 GB checkpoint 只要幾分鐘）。網路磁碟不允許 `chown`，rsync 的警告可忽略。
- RunPod 的 API key 只在建立當下顯示完整內容，列表頁是遮罩；`RUNPOD_POD_ID` 在 ssh 進來的 shell 沒有，
  從 `/proc/1/environ` 取。守護程序裡把 pod ID 寫死。
- 一次 pod（約 6 小時、US$3）跑完：A18 三個資料集臂共 21 個 run、153＋44 個分解。

## A23（A18 推到 ImageNet-1k，ResNet-18；plan §9.11）—— 2026-09-11 準備

租機：**單張 4090、vCPU 越多越好（≥ 24）、容器磁碟 ≥ 120 GB 或 /workspace 網路磁碟 ≥ 100 GB**
（JPEG 45 GB＋每個 worker 一個 1 GB 的 parquet 暫存＋六個 run 的 checkpoint 約 1 GB）。
估：抓資料 1–2 h；每個 run 90 epoch，兩個並行時每 epoch 約 15–20 min → 一對約 24–30 h；三對約 3–3.5 天。
預算 US$70–90（作者 09-10 核准）。

```bash
# 進 pod 之後（程式碼照第 3 節送上去；HF token 要先登入，資料集是 gated）
pip install huggingface_hub pyarrow            # 其他套件同 A18
hf auth login                                   # 貼 Mac 上 ~/.cache/huggingface/token 的內容
cd /workspace/effective-width/code
# 0. 前置：CIFAR-10 ResNet-18 三臂 x 3 種子（約 30–40 min，可與抓資料同時）
PAR=3 nohup bash scripts/run_a18_r18_cifar_pod.sh > ../results/a18_r18.log 2>&1 &
# 1. 主實驗：抓資料 + 六個 run，全自動排序
DATA=/workspace/imagenet nohup bash scripts/run_a18_imagenet_pod.sh > ../results/a18_imagenet.log 2>&1 &
# 進度
tail -f ../results/a18_imagenet.log ../results/a18_imagenet/resnet18_full_s0.log
```

守護程序（跑完 commit、push、關機；REST key 先放 `/root/.runpod/restkey`）：
```bash
nohup bash -c 'until grep -q "A18-ImageNet done" results/a18_imagenet.log; do sleep 600; done; git add results/a18_imagenet results/a18_r18 results/a18_imagenet.log results/a18_r18.log && git commit -q -m "A23: ResNet-18 ImageNet-1k arms + CIFAR-10 preflight from the pod" && git fetch -q origin && git rebase origin/master && git push origin master; curl -s -X POST https://rest.runpod.io/v1/pods/<POD_ID>/stop -H "Authorization: Bearer $(cat /root/.runpod/restkey)"' > results/a23_autostop.log 2>&1 &
```
`.pt` 不進 git；六個 `final.pt`（各 45 MB）與兩個全寬 `final.pt` 跑完後用 rsync 拉回 Mac
`results/a18_imagenet_pt/`（之後量剖面、分解用）。

- 前置若 (e) 比 (a) 掉超過 1 點：先停主實驗（`pkill -f run_a18_imagenet` 前先 `ps` 確認，見 09-10 筆記），
  查 `results/a18_r18/widths_act.json` 的 `sites`（stage max 規則）再決定。
- 中途斷線／pod 重啟：同一條指令重跑即可，每個 run 由 `resume.pt` 續跑，抓資料由 `.done/` 標記略過。
- 兩個 run 並行時 `WORKERS` 預設 nproc/2；若 `img/s` 明顯低於 1,200，是解碼不夠，換 vCPU 多的機型。

### 2026-09-11 實跑筆記（A23 pod：4090、頁面寫 12 vCPU／31 GB，實際 cgroup 13.6 核／61 GB，US$0.74/h）

- **網路碟（/workspace，MooseFS）不能放小檔資料集**：12 個程序隨機開檔只有 1,430 檔/s（冷、暖都一樣），
  一個 ResNet-18 run 就要 1,500 img/s。同一批圖串成一個大檔後隨機 seek+read 有 5,600 img/s（冷）、29,000（暖）。
  所以 `fetch_imagenet.py` 預設 `--format blob`（每個 parquet shard 一個 .blob＋.idx.npy），
  `scripts/imagenet_blob.py` 的 `BlobImageFolder` 讀。容器磁碟只有 30 GB，放不下 32 GB 的訓練集。
- 抓 ImageNet-1k：HF 294 個 train shard（146 GB）＋14 個 val shard，12 個 worker 邊下載邊轉，**8 分鐘**全部完成
  （這個機房對 HF 的頻寬很大）。轉完的訓練集 32 GB、驗證集 1.3 GB。
- **API key 權限**：Restricted 只勾 `api.runpod.ai` 的 key **打不到** `rest.runpod.io`（那是 Serverless 端點的主機），
  停 pod 要 GraphQL Read/Write，等於 All。建 key 後一定先用 `GET /v1/pods/<id>` 驗證（HTTP 200）再裝守護程序。
- pod 自己的守護程序：`/workspace/autostop.sh`（兩條隊列都 done、或 ImageNet runner 提前退出時 stop）。
  結果在 /workspace，stop 後仍在；之後從 Mac rsync `results/a18_imagenet`、`results/a18_r18` 回來再 commit。
- 兩個 ImageNet run 並行＋CIFAR 前置同時跑時每個 run 約 960 img/s（GPU 100%）；前置跑完後再量。
- HF token 直接 scp Mac 的 `~/.cache/huggingface/token` 到 pod 的 `/root/.cache/huggingface/token` 即可，不用 `hf auth login`。

### A23 週一回收步驟（2026-09-11 寫；pod `dr7qq4pkwtnpri`，ssh `root@213.173.109.33 -p 19311 -i ~/.ssh/id_ed25519`）

1. RunPod 控制台看 pod 狀態。**Stopped** = 守護程序已停機（正常）；**Running** = 還沒跑完或守護程序死了，先 ssh 看
   `tail /workspace/results/a23_autostop.log` 與 `grep "^  epoch" /workspace/results/a18_imagenet/resnet18_*_s0.log | tail`。
2. 若 Stopped：在控制台 **Start** 同一個 pod（容器磁碟會重置，/workspace 不會；ssh port 可能變，重抓）。
   若是被 RunPod 收回而中途停的：Start 後重跑 `cd /workspace/code && DATA=/workspace/imagenet WORKERS=6 nohup bash scripts/run_a18_imagenet_pod.sh >> ../results/a18_imagenet.log 2>&1 &`
   ，每個 run 從 `resume.pt` 續跑；再 `bash /workspace/start_autostop.sh`（restkey 在容器磁碟，重置後要重貼）。
3. 拉結果（小檔）：
   `rsync -a --exclude '*.pt' --exclude '.parquet' -e "ssh -p <port> -i ~/.ssh/id_ed25519" root@<host>:/workspace/results/a18_imagenet root@<host>:/workspace/results/a18_imagenet.log root@<host>:/workspace/results/a23_autostop.log results/`
   拉權重（8 個 final.pt 各 45 MB，之後量剖面／分解用）：
   `rsync -a --include '*/' --include 'final.pt' --exclude '*' -e "ssh -p <port> -i ~/.ssh/id_ed25519" root@<host>:/workspace/results/a18_imagenet/ results/a18_imagenet_pt/`
4. 判定：`cd code && ~/.venvs/effwidth/bin/python scripts/checkpoint_analysis.py --results ../results | sed -n '/^24\./,$p'`
   → plan §10 新增 10.10（P9.29／P9.30 照字面判），依 plan §9.11 後果改 `sec:a18`（ImageNet 段＋表列）、§VI-C、摘要，
   然後做投稿門檻 C1（§I 的 so-what 提前）與 C2。
5. 確認 pod 已 Stop 後，到 RunPod → Storage 看網路磁碟是否要 Terminate（留著每月約 US$5；資料集可以重抓 8 分鐘，
   建議權重拉回 Mac 後 Terminate）。**刪掉 API key `a23-autostop`**（All 權限）。
