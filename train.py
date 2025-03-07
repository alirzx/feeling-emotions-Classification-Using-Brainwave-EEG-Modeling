import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
import pandas as pd

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """
    ایجاد DataLoader برای مجموعه‌های آموزش، اعتبارسنجی و تست
    """
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    آموزش یک دوره کامل
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_X, batch_y in tqdm(train_loader, desc="Training"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # جمع‌آوری پیش‌بینی‌ها و برچسب‌ها
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
    
    # محاسبه معیارها
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, metrics

def validate(model, val_loader, criterion, device):
    """
    اعتبارسنجی مدل
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(val_loader, desc="Validating"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            
            # جمع‌آوری پیش‌بینی‌ها و برچسب‌ها
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # محاسبه معیارها
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, metrics

def plot_metrics(history, test_metrics, model_name="EEG-Emotion-CNN"):
    """
    رسم نمودارهای معیارهای آموزش و تست
    """
    plt.figure(figsize=(15, 10))
    
    # نمودار loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # نمودار accuracy
    plt.subplot(2, 2, 2)
    train_acc = [m['accuracy'] for m in history['train_metrics']]
    val_acc = [m['accuracy'] for m in history['val_metrics']]
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # ماتریس درهم‌ریختگی
    plt.subplot(2, 2, 3)
    cm = test_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # نمودار ROC
    plt.subplot(2, 2, 4)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = cm.shape[0]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_metrics['y_true'] == i, test_metrics['y_prob'][:, i])
        roc_auc[i] = roc_auc_score(test_metrics['y_true'] == i, test_metrics['y_prob'][:, i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_metrics.png')
    plt.close()

def calculate_metrics(y_true, y_pred, y_prob):
    """
    محاسبه معیارهای ارزیابی
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob, multi_class='ovr'),
        'y_true': y_true,
        'y_prob': y_prob
    }
    return metrics

def print_detailed_metrics(test_metrics, history):
    """
    چاپ معیارهای جزئی و ذخیره آنها در فایل
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    
    print("\n1. FINAL TEST METRICS:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    print("\n2. CLASSIFICATION REPORT:")
    print(test_metrics['classification_report'])
    
    print("\n3. PER-CLASS METRICS:")
    cm = test_metrics['confusion_matrix']
    n_classes = cm.shape[0]
    
    for i in range(n_classes):
        print(f"\nClass {i}:")
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
        
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1-Score: {f1:.4f}")
    
    # ذخیره نتایج
    results_dict = {
        'test_accuracy': test_metrics['accuracy'],
        'test_auc_roc': test_metrics['auc_roc'],
        'classification_report': test_metrics['classification_report'],
        'confusion_matrix': test_metrics['confusion_matrix'].tolist()
    }
    
    pd.DataFrame([results_dict]).to_csv('model_metrics.csv', index=False)
    print("\nMetrics have been saved to 'model_metrics.csv'")
    
    return results_dict 