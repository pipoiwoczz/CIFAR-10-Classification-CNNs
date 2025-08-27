import torchvision as tv
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
from .config import CFG

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]

def get_transforms():
    train_tf = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MEAN, STD),
    ])
    val_tf = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf

def load_full_train_without_transform(data_root):
    return tv.datasets.CIFAR10(root=data_root, train=True, download=True, transform=None)

class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class SubsetWithTransform(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset          # must have transform=None
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]   # returns (PIL, int)
        if self.transform is not None:
            x = self.transform(x)             # -> Tensor (C,H,W)
        return x, y


def make_fold_loaders(base_train, train_idx, val_idx, cfg):
    train_tf = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MEAN, STD),
    ])
    val_tf = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MEAN, STD),
    ])

    ds_train = SubsetWithTransform(base_train, train_idx, transform=train_tf)
    ds_val   = SubsetWithTransform(base_train, val_idx,   transform=val_tf)

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False, persistent_workers=False)
    val_loader   = DataLoader(ds_val,   batch_size=cfg.batch_size*2, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=False, persistent_workers=False)
    
    return train_loader, val_loader

def get_test_loader(cfg : CFG, data_root: str) -> DataLoader:
    test_tf = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MEAN, STD),
    ])
    test_dataset = tv.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size*2, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=False, persistent_workers=False)
    return test_loader