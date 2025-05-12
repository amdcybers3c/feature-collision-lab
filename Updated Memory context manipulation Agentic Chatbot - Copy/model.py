import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ------------------------
# Model & Dataset Classes
# ------------------------

class AttackDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AttackDetectionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class AttackDetectionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ------------------------
# Model Utilities
# ------------------------

def load_model():
    model_path = "attack_detection_model.pth"
    input_size = 10
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = AttackDetectionModel(input_size, hidden_size, num_layers, output_size)

    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception:
        try:
            model.load_state_dict(torch.load(model_path, weights_only=False))
        except Exception:
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
            model.load_state_dict(torch.load(model_path))

    model.eval()

    metrics = None
    loss_history = None

    if os.path.exists("model_metrics.pt"):
        try:
            metrics = torch.load("model_metrics.pt", weights_only=True)
        except:
            try:
                metrics = torch.load("model_metrics.pt", weights_only=False)
            except:
                torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
                metrics = torch.load("model_metrics.pt")

    if os.path.exists("training_losses.csv"):
        loss_history = pd.read_csv("training_losses.csv")

    return model, metrics, loss_history

def train_model(force_retrain=False):
    """Train the model and save metrics"""
    model_path = "attack_detection_model.pth"

    if os.path.exists(model_path) and not force_retrain:
        return load_model()

    # Generate synthetic data
    def generate_synthetic_data(n=1000):
        X = np.random.rand(n, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=(n, 1)).astype(np.float32)
        return X, y

    X, y = generate_synthetic_data(2000)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_data = torch.tensor(X_train).unsqueeze(1)
    val_data = torch.tensor(X_val).unsqueeze(1)
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    train_loader = DataLoader(AttackDetectionDataset(train_data, train_labels), batch_size=32)
    val_loader = DataLoader(AttackDetectionDataset(val_data, val_labels), batch_size=32)

    model = AttackDetectionModel(10, 64, 2, 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                loss = criterion(output, y)
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(val_loader))

    torch.save(model.state_dict(), model_path)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            output = model(x)
            preds = (output >= 0.5).float()
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(y.numpy().flatten())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds) * 100,
        "recall": recall_score(all_labels, all_preds) * 100,
        "confusion_matrix": confusion_matrix(all_labels, all_preds)
    }

    torch.save(metrics, "model_metrics.pt")

    pd.DataFrame({
        "epoch": list(range(1, len(train_losses)+1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    }).to_csv("training_losses.csv", index=False)

    return model, metrics, pd.read_csv("training_losses.csv")

# ------------------------
# Inference Utility
# ------------------------

def evaluate_text(model, text):
    """Evaluate a conversation for attack patterns"""
    conversation = [msg for msg in text.split('\n') if msg.strip()]
    text_full = " ".join(conversation)

    avg_msg_length = np.mean([len(msg) for msg in conversation])
    max_msg_length = max([len(msg) for msg in conversation])
    total_length = len(text_full)

    admin_keywords = ['admin', 'administrator', 'root', 'superuser', 'privileged', 'access']
    identity_keywords = ['i am', 'my role', 'my access', 'my name', 'my id']
    system_keywords = ['system', 'database', 'credentials', 'password', 'token', 'auth']

    admin_count = sum(1 for word in admin_keywords if word in text_full.lower())
    identity_count = sum(1 for phrase in identity_keywords if phrase in text_full.lower())
    system_count = sum(1 for word in system_keywords if word in text_full.lower())

    positive_words = ['please', 'help', 'thanks', 'appreciate', 'good']
    negative_words = ['wrong', 'mistake', 'error', 'incorrect', 'bad']

    positive_count = sum(1 for word in positive_words if word in text_full.lower())
    negative_count = sum(1 for word in negative_words if word in text_full.lower())

    question_ratio = text_full.count('?') / (len(conversation) + 0.1)
    command_ratio = sum(1 for msg in conversation if msg.strip().endswith('.')) / (len(conversation) + 0.1)

    features = [
        avg_msg_length / 100,
        max_msg_length / 500,
        total_length / 2000,
        admin_count / 10,
        identity_count / 10,
        system_count / 10,
        positive_count / 10,
        negative_count / 10,
        question_ratio,
        command_ratio
    ]

    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    probability = output.item()
    prediction = 1 if probability >= 0.5 else 0

    return prediction, probability

