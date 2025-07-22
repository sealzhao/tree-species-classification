import numpy as np
from sklearn.model_selection import train_test_split
from trainer.train import train_model

if __name__ == "__main__":
    config = {
        "input_dim": 1440,
        "model_dim": 384,
        "num_heads": 4,
        "num_layers": 2,
        "num_classes": 19,
        "batch_size": 64,
        "lr": 1e-4,
        "epochs": 100,
        "save_path": "checkpoints/best_model.pth"
    }


    X = np.load("data/X_1440.npy")
    y = np.load("data/y_l3_1440.npy")

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    train_model(X_train, y_train, X_val, y_val, config)
