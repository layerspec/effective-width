#!/bin/bash
# A20 datasets (plan 9.9): STL-10, SVHN via torchvision; CIFAR-10-C from Zenodo.
cd /Users/ccli/Downloads/effective-width-claude
~/.venvs/effwidth/bin/python -c "
import torchvision
torchvision.datasets.STL10('data', split='train', download=True); torchvision.datasets.STL10('data', split='test', download=True)
torchvision.datasets.SVHN('data/svhn', split='train', download=True); torchvision.datasets.SVHN('data/svhn', split='test', download=True)
print('stl10 + svhn ok')"
if [ ! -f data/CIFAR-10-C/labels.npy ]; then
  mkdir -p data && curl -L -C - -o data/CIFAR-10-C.tar "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1" && tar -xf data/CIFAR-10-C.tar -C data && rm -f data/CIFAR-10-C.tar
fi
ls data/CIFAR-10-C | head -3; echo "=== a20 downloads done $(date) ==="
