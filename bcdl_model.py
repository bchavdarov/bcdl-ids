import torch.nn as nn


class BCSimpleCNN(nn.Module):
    def __init__(self, num_classes=2, conv1_out=32):
        super().__init__()
        c1, c2, c3 = conv1_out, conv1_out * 2, conv1_out * 4

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),  # works for 32x32, 224x224, etc.
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
