from torchvision.transforms import ToTensor, Normalize, Resize, Compose
import numpy as np
import albumentations as A


class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        image = np.array(image)
        augmented = self.transform(image=image)
        return augmented['image']

def return_transform():
    train_transform = AlbumentationsWrapper(A.Compose([
        A.RandomCrop(28, 28),
        A.Rotate(limit=20, p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        A.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
        A.ToTensorV2()
    ]))

    test_transform = Compose([
        Resize((28, 28)),
        ToTensor(),
        Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970))
    ])

    return train_transform, test_transform