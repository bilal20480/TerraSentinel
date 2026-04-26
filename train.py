import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ===== LOAD DATA (correct header row) =====
gw = pd.read_excel("groundwater.xlsx", header=3)
lulc = pd.read_excel("lulc.xlsx", header=3)

# ===== CLEAN COLUMN NAMES =====
gw.columns = gw.columns.str.strip()
lulc.columns = lulc.columns.str.strip()

# ===== PARSE DATES =====
gw["Date"] = pd.to_datetime(gw["Date"], errors="coerce")
lulc["Date"] = pd.to_datetime(lulc["Date"], errors="coerce")

# ===== FIND GW COLUMN AUTOMATICALLY =====
gw_col = [c for c in gw.columns if "GW Anomaly" in c][0]

# ===== CLEAN GW =====
gw = gw[gw[gw_col] != "—"]
gw[gw_col] = pd.to_numeric(gw[gw_col], errors="coerce")

# ===== CLEAN LULC =====
if "Flag" in lulc.columns:
    lulc = lulc[lulc["Flag"] != "LOW DATA"]

lulc_features = ["Water", "Trees", "Grass", "Crops", "Built-up", "Bare"]
lulc = lulc[["Date"] + lulc_features]

# Convert to numeric safely
for col in lulc_features:
    lulc[col] = pd.to_numeric(lulc[col], errors="coerce")

# Normalize LULC
lulc[lulc_features] = lulc[lulc_features].div(
    lulc[lulc_features].sum(axis=1), axis=0
)

# ===== MERGE =====
data = pd.merge(gw, lulc, on="Date", how="inner")
data = data.sort_values("Date")
data = data.dropna()

# ===== CHECK DATA SIZE =====
if len(data) < 10:
    print("⚠ Very small dataset after cleaning:", len(data))

# ===== FEATURES =====
X = data[lulc_features]
y = data[gw_col]

# ===== SCALE =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== SEQUENCES =====
SEQ_LEN = 3   # smaller to avoid empty dataset
X_seq, y_seq = [], []

for i in range(len(X_scaled) - SEQ_LEN):
    X_seq.append(X_scaled[i:i+SEQ_LEN])
    y_seq.append(y.iloc[i+SEQ_LEN])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)

# ===== SAFETY CHECK =====
if len(X_seq) == 0:
    raise ValueError("❌ Not enough data to create sequences. Reduce SEQ_LEN or check data.")

# ===== TENSORS =====
X_tensor = torch.tensor(X_seq).float()
y_tensor = torch.tensor(y_seq).float()

# ===== MODEL =====
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = LSTMModel(input_size=X_tensor.shape[2])

# ===== TRAIN =====
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    preds = model(X_tensor).squeeze()
    loss = loss_fn(preds, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# ===== EVALUATE =====
model.eval()
with torch.no_grad():
    preds = model(X_tensor).squeeze().numpy()

preds = np.array(preds).flatten()
y_true = np.array(y_seq).flatten()

if len(preds) > 1:
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    print("Final RMSE:", rmse)
else:
    print("⚠ Not enough samples to compute RMSE")

# ===== SAVE =====
torch.save(model.state_dict(), "lulc_gw_model.pth")
print("✅ Model saved as lulc_gw_model.pth")