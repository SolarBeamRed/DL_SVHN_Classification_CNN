from src.utils.config import DATA_DIR
from src.data.build_transforms import return_transform
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset

def load_data():
    #Datasets
    train_transform, test_transform = return_transform()
    train_dataset = SVHN(root=DATA_DIR, split='train', transform=train_transform, download=True)
    val_dataset = SVHN(root=DATA_DIR, split='train', transform=test_transform, download=True)
    test_dataset = SVHN(root=DATA_DIR, split='test', transform=test_transform, download=True)

    train_size = int(0.6 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_dataset = Subset(train_dataset, train_subset.indices)
    val_dataset = Subset(val_dataset, val_subset.indices)

    extra_dataset = SVHN(root=DATA_DIR, split='extra', transform=train_transform, download=True)

    train_dataset = ConcatDataset([train_dataset, extra_dataset])
    del train_subset
    del val_subset
    #____________________________________________________________________________________________

    #Loaders
    loader_train = DataLoader(dataset=train_dataset, shuffle=True, batch_size=512, pin_memory=True,
                              in_order=False, prefetch_factor=3, num_workers=4, persistent_workers=True)

    loader_test = DataLoader(dataset=test_dataset, batch_size=512, pin_memory=True,
                             in_order=False, prefetch_factor=3, num_workers=4, persistent_workers=True)

    loader_val = DataLoader(dataset=val_dataset, batch_size=512, pin_memory=True, in_order=False,
                            prefetch_factor=3, num_workers=4, persistent_workers=True)

    return loader_train, loader_val, loader_test