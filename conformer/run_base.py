import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from dataloader import LibriSpeechDataset, custom_collate_fn
from model import Conformer
from tqdm import tqdm
import matplotlib.pyplot as plt

# Use CUDA if available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
batch_size = 32
num_epochs = 10
num_classes = 29  # 26 letters + space + apostrophe + blank token for CTC
input_dim = 80
encoder_dim = 256
num_encoder_layers = 4
model_save_path = "./saved_models_new"  # Directory to save the model
plot_save_path = "./plots_new"  # Directory to save the plots

# Create directories if they don't exist
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(plot_save_path, exist_ok=True)

# Initialize dataset and dataloader
root_dirs = [
    "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/train-clean-100",
    "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/train-clean-360",
    "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/train-other-500",
]

dataset = LibriSpeechDataset(root_dirs=root_dirs, ratio = 0.2)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=4)

# Initialize MelSpectrogram transformation
mel_transform = MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=input_dim
).to(device)

# Initialize the Conformer model
model = Conformer(
    num_classes=num_classes,
    input_dim=input_dim,
    encoder_dim=encoder_dim,
    num_local_layers=0,
    num_encoder_layers=num_encoder_layers,
).to(device)

# Save model architecture summary
with open(os.path.join(model_save_path, "model_baseline_summary.txt"), "w") as f:
    f.write(str(model))

# Evaluation metric: Word Error Rate (WER)
def calculate_wer(hypotheses, references):
    def edit_distance(h, r):
        dp = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
        for i in range(len(h) + 1):
            dp[i][0] = i
        for j in range(len(r) + 1):
            dp[0][j] = j
        for i in range(1, len(h) + 1):
            for j in range(1, len(r) + 1):
                if h[i - 1] == r[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[-1][-1]

    total_distance = 0
    total_words = 0
    for hyp, ref in zip(hypotheses, references):
        total_distance += edit_distance(hyp, ref)
        total_words += len(ref)
    return total_distance / total_words if total_words > 0 else float("inf")

# Training Metrics
epoch_wer = []
epoch_loss = []

# Training Loop
print("Starting Training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_wer = 0
    total_batches = 0
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, (padded_waveforms, sample_rates, transcripts) in enumerate(dataloader):
            try:
                # Prepare inputs and targets
                mel_inputs = mel_transform(padded_waveforms.to(device).squeeze(1)).transpose(1, 2).to(device)  # [B, T, 80]
                input_lengths = torch.full((mel_inputs.size(0),), mel_inputs.size(1), dtype=torch.long, device=device)

                # Forward pass
                outputs, output_lengths = model(mel_inputs, input_lengths)
                hypotheses = torch.argmax(outputs, dim=-1).cpu().numpy()
                references = [list(t.lower()) for t in transcripts]

                # Calculate Word Error Rate
                wer = calculate_wer(hypotheses, references)
                total_wer += wer
                total_batches += 1

                # Update progress bar
                pbar.set_postfix(WER=wer)
                pbar.update(1)

            except Exception as e:
                print(f"RuntimeError during forward pass: {e}")
                continue

    # Average metrics for the epoch
    average_wer = total_wer / total_batches
    epoch_wer.append(average_wer)
    print(f"Average WER for Epoch {epoch + 1}: {average_wer:.4f}")

    # Save model checkpoint after each epoch
    model_file = os.path.join(model_save_path, f"conformer_baseline_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}\n")

# Plot WER
plt.figure()
plt.plot(range(1, num_epochs + 1), epoch_wer, label="WER")
plt.xlabel("Epoch")
plt.ylabel("Word Error Rate (WER)")
plt.title("WER Across Epochs")
plt.legend()
plt.savefig(os.path.join(plot_save_path, "wer_baseline_plot.png"))
plt.close()

print("Training Complete!")
print(f"Plots saved to {plot_save_path}")
