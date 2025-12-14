import torch
import torch.nn as nn
import numpy as np
import joblib


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # ambil output timestep terakhir
        out = self.fc(out)
        return out


def train_lstm_torch(df, save_model="model_lstm_torch.pth",
                     save_scaler="lstm_scaler.pkl",
                     seq_len=14, epochs=40):

    from sklearn.preprocessing import MinMaxScaler

    close = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(seq_len, len(close_scaled)):
        X.append(close_scaled[i - seq_len:i, 0])
        y.append(close_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    model = LSTMPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_model)
    joblib.dump({"scaler": scaler, "seq_len": seq_len}, save_scaler)

    return {"loss": float(loss.item())}


def predict_lstm_torch(df, model_path="model_lstm_torch.pth",
                       scaler_path="lstm_scaler.pkl"):

    rec = joblib.load(scaler_path)
    scaler = rec["scaler"]
    seq_len = rec["seq_len"]

    model = LSTMPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    close = df["Close"].values.reshape(-1, 1)
    close_scaled = scaler.transform(close)

    last_seq = close_scaled[-seq_len:].reshape(1, seq_len, 1)
    last_seq = torch.tensor(last_seq, dtype=torch.float32)

    pred_scaled = model(last_seq).item()
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]

    return pred
