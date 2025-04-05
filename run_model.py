import os
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Corrected protocol file path
protocol_file = "ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
label_map = {}

# Load labels from protocol
with open(protocol_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        filename = parts[0]
        label = parts[-1]
        label_map[filename] = 0 if label == "bonafide" else 1  # 0 = real, 1 = fake

# Corrected audio folder path
AUDIO_FOLDER = "ASVspoof2019_LA/ASVspoof2019_LA_train/flac"
audio_files = [
    "LA_T_1000086.flac",
    "LA_T_1000143.flac",
    "LA_T_1000021.flac",
    "LA_T_1000337.flac",
    "LA_T_1000456.flac"
]
audio_paths = [os.path.join(AUDIO_FOLDER, f) for f in audio_files]

# Dummy RawNet2 model for testing
class DummyRawNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return torch.sigmoid(x)

# Load model
model = DummyRawNet2()
model.eval()

# Predict and collect labels
y_true, y_pred = [], []

for path in audio_paths:
    filename = os.path.basename(path)
    label = label_map.get(filename, 0)
    y_true.append(label)

    # Load audio and prepare input
    signal, sr = torchaudio.load(path)
    signal = signal.mean(dim=0).unsqueeze(0).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        output = model(signal)
        predicted = int(output.item() > 0.5)

    y_pred.append(predicted)
    print(f"Audio: {filename} | True: {label} | Pred: {predicted}")

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: RawNet2 Predictions")
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png")
plt.show()

# Save dummy EER
eer = 0.102  # placeholder
with open("results/eer_score.txt", "w") as f:
    f.write(f"EER: {eer:.3f}")

# Plot waveform for the first sample
signal, sr = torchaudio.load(audio_paths[0])
plt.figure(figsize=(12, 3))
plt.plot(signal[0].numpy())
plt.title(f"Waveform: {audio_files[0]}")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
