import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from model import EEGNet
from train import (
    create_data_loaders,
    train_epoch,
    validate,
    plot_metrics,
    print_detailed_metrics
)

def main():
    # تنظیمات اولیه
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # پارامترهای آموزش
    batch_size = 32
    n_epochs = 100
    learning_rate = 0.001
    patience = 10
    
    # بارگذاری داده‌ها
    # این بخش باید با داده‌های واقعی شما جایگزین شود
    X = np.load('data/X.npy')  # داده‌های ورودی
    y = np.load('data/y.npy')  # برچسب‌ها
    
    # پیش‌پردازش داده‌ها
    scaler = StandardScaler()
    le = LabelEncoder()
    
    X_scaled = scaler.fit_transform(X)
    y_encoded = le.fit_transform(y)
    
    # تقسیم داده‌ها
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # ایجاد DataLoaderها
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size
    )
    
    # ایجاد مدل
    input_size = X_train.shape[1]
    n_classes = len(np.unique(y))
    model = EEGNet(input_size, n_classes).to(device)
    
    # تابع هزینه و بهینه‌ساز
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # آموزش مدل
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(n_epochs):
        # آموزش
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # اعتبارسنجی
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # به‌روزرسانی نرخ یادگیری
        scheduler.step(val_loss)
        
        # ذخیره تاریخچه
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # چاپ پیشرفت
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # ذخیره بهترین مدل
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'history': history,
                'class_labels': le.classes_
            }, 'eeg_emotion_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
    
    # ارزیابی نهایی
    print("\nPerforming final evaluation...")
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    print_detailed_metrics(test_metrics, history)
    plot_metrics(history, test_metrics, model_name="EEG-Emotion-CNN")

if __name__ == "__main__":
    main() 