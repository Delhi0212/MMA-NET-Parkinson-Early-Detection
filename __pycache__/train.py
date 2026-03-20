import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from preprocessing import get_dataloaders


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
LR = 0.0003

print("Using device:", DEVICE)


# ==========================================
# VOICE TRANSFORMER
# ==========================================

class VoiceTransformer(nn.Module):

    def __init__(self, num_features=22, d_model=64):

        super().__init__()

        self.embedding = nn.Linear(1, d_model)

        self.pos = nn.Parameter(torch.randn(1, num_features, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, x):

        x = x.unsqueeze(-1)

        x = self.embedding(x)

        x = x + self.pos[:, :x.size(1), :]

        x = self.transformer(x)

        return x


# ==========================================
# GAIT TRANSFORMER
# ==========================================

class GaitTransformer(nn.Module):

    def __init__(self, seq_len=300, input_dim=16, d_model=64):

        super().__init__()

        self.projection = nn.Linear(input_dim, d_model)

        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, x):

        x = self.projection(x)

        x = x + self.pos[:, :x.size(1), :]

        x = self.transformer(x)

        return x


# ==========================================
# MMA NET
# ==========================================

class MMANet(nn.Module):

    def __init__(self):

        super().__init__()

        self.voice = VoiceTransformer()

        self.gait = GaitTransformer()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )

        self.classifier = nn.Sequential(

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)

        )

    def forward(self, v, g):

        v_feat = self.voice(v)

        g_feat = self.gait(g)

        v_summary = v_feat.mean(dim=1)

        query = v_summary.unsqueeze(1)

        attn_out, _ = self.cross_attn(
            query=query,
            key=g_feat,
            value=g_feat
        )

        attn_out = attn_out.squeeze(1)

        combined = torch.cat([v_summary, attn_out], dim=1)

        return self.classifier(combined)


# ==========================================
# TRAINING
# ==========================================

def train():

    train_loader, test_loader = get_dataloaders()

    model = MMANet().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion = nn.CrossEntropyLoss()

    print("\nTraining Started\n")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for batch in train_loader:

            voice = batch["voice"].to(DEVICE)

            gait = batch["gait"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(voice, gait)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "mma_net_model.pth")

    print("\nModel Saved")

    evaluate(model, test_loader)


# ==========================================
# EVALUATION
# ==========================================

def evaluate(model, test_loader):

    model.eval()

    preds_all = []
    labels_all = []

    with torch.no_grad():

        for batch in test_loader:

            voice = batch["voice"].to(DEVICE)

            gait = batch["gait"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            outputs = model(voice, gait)

            preds = torch.argmax(outputs, dim=1)

            preds_all.extend(preds.cpu().numpy())

            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)

    f1 = f1_score(labels_all, preds_all)

    print("\nRESULTS")

    print("Accuracy:", acc * 100)

    print("F1 Score:", f1 * 100)

    print("\nConfusion Matrix")

    print(confusion_matrix(labels_all, preds_all))

    print("\nClassification Report")

    print(classification_report(labels_all, preds_all))


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    train()