import torch.nn as nn

class TunedModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.05),

            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(256, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, X):
        X = self.features(X)
        return self.classifier(X)