"""Top-1 of the five re-measured weights on 320 validation images under the ImageNet statistics (as first measured)
and under each weight's own pretrained_cfg statistics (as re-measured).  Labels from the wnid in the file name."""
import glob, os, sys
import torch, timm
from PIL import Image
from timm.data import ImageNetInfo
import torchvision.transforms as T

torch.set_num_threads(4)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
files = sorted(glob.glob("../data/imagenet_val_6400/*.JPEG"))[::20][:320]
info = ImageNetInfo()
wn = {w: i for i, w in enumerate(info.label_names())}
y = torch.tensor([wn[os.path.basename(f).rsplit("_", 1)[1].split(".")[0]] for f in files])
geo = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
x = torch.stack([geo(Image.open(f).convert("RGB")) for f in files])
IMN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
for name in ["mobilenetv3_large_100.ra4_e3600_r224_in1k", "mobilenetv3_large_100.miil_in21k_ft_in1k",
             "tf_mobilenetv3_large_100.in1k", "efficientnet_b0.ra4_e3600_r224_in1k", "tf_efficientnet_b0.ap_in1k"]:
    m = timm.create_model(name, pretrained=True).eval()
    own = (tuple(m.pretrained_cfg["mean"]), tuple(m.pretrained_cfg["std"]))
    out = []
    for mean, std in (IMN, own):
        mu, sd = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)
        with torch.no_grad():
            p = torch.cat([m((x[i:i + 64] - mu) / sd).argmax(1) for i in range(0, len(x), 64)])
        out.append(100 * (p == y).float().mean().item())
    print(f"{name:45s} own stats {own[0]} / {own[1]}   top-1 ImageNet stats {out[0]:5.1f}   own stats {out[1]:5.1f}", flush=True)
