import torch
import torch.nn as nn

class TreeTransformer(nn.Module):
    def __init__(self, input_dim=1440, model_dim=256, num_heads=4, num_layers=2, num_classes=19, dropout=0.1):
        super(TreeTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (B, input_dim)
        x = self.embedding(x).unsqueeze(1)  # → (B, 1, model_dim)
        x = self.transformer(x)             # → (B, 1, model_dim)
        x = x[:, 0, :]                      # 取第一个 token
        return self.classifier(x)
