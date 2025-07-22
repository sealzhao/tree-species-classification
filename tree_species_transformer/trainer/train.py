import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from models.transformer import TreeTransformer
from dataset.tree_dataset import TreeSpeciesDataset
from collections import Counter
import numpy as np

def get_class_weights(y, num_classes):
    counter = Counter(y)
    freqs = np.array([counter[i] if i in counter else 1 for i in range(num_classes)])
    weights = 1.0 / (np.log(freqs + 1.0))
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)

def train_model(X_train, y_train, X_val, y_val, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)

    train_set = TreeSpeciesDataset(X_train, y_train)
    val_set = TreeSpeciesDataset(X_val, y_val)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

    model = TreeTransformer(
        input_dim=config["input_dim"],
        model_dim=config["model_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    class_weights = get_class_weights(y_train, config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        print(f"\n[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                out = model(X_batch)
                preds = out.argmax(dim=1).cpu()
                all_preds.extend(preds)
                all_labels.extend(y_batch)

        acc = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {acc:.4f}")
        print(classification_report(all_labels, all_preds, digits=3, zero_division=0))

        # 保存当前轮次模型
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

        # 保存 best 模型
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), config["save_path"])
            print(f"💾 Best model updated (accuracy={acc:.4f})")

    print(f"\n✅ Training complete. Best Val Accuracy: {best_val_acc:.4f}")
