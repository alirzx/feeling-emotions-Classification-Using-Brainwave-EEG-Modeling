import torch
import torch.nn as nn

class EEGEmotionClassifier(nn.Module):
    def __init__(self, input_size=2548, num_classes=4):
        super(EEGEmotionClassifier, self).__init__()
        
        # لایه‌های شبکه عصبی
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
        )
        
        # لایه‌های طبقه‌بندی
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

def create_model(input_size=2548, num_classes=4):
    """
    ایجاد نمونه‌ای از مدل با پارامترهای مشخص شده
    """
    model = EEGEmotionClassifier(input_size=input_size, num_classes=num_classes)
    return model

if __name__ == "__main__":
    # تست مدل
    model = create_model()
    x = torch.randn(32, 2548)  # یک نمونه تصادفی با اندازه batch=32
    output = model(x)
    print(f"شکل خروجی: {output.shape}")  # باید (32, 4) باشد 