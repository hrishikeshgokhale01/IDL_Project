import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class LibriSpeechDatasetWithNoise(Dataset):
    """
    Custom Dataset for the LibriSpeech dataset with optional noise injection.
    Traverses directories to load audio files and corresponding transcripts.
    """
    def __init__(self, root_dirs, transform=None, ratio=1.0, noise_type=None, noise_intensity=0.01):
        self.root_dirs = root_dirs
        self.transform = transform
        self.noise_type = noise_type
        self.noise_intensity = noise_intensity
        self.samples = []
        
        # Traverse each root directory (train, test, dev)
        for root_dir in self.root_dirs:
            for speaker in os.listdir(root_dir):
                speaker_dir = os.path.join(root_dir, speaker)
                if not os.path.isdir(speaker_dir):
                    continue
                
                for chapter in os.listdir(speaker_dir):
                    chapter_dir = os.path.join(speaker_dir, chapter)
                    if not os.path.isdir(chapter_dir):
                        continue
                    
                    # Load the transcript file
                    transcript_file = os.path.join(chapter_dir, f"{speaker}-{chapter}.trans.txt")
                    if not os.path.exists(transcript_file):
                        continue
                    
                    with open(transcript_file, 'r') as f:
                        transcripts = {line.split()[0]: " ".join(line.split()[1:]) for line in f}
                    
                    # Limit the number of files processed by the ratio
                    audio_files = [f for f in os.listdir(chapter_dir) if f.endswith(".flac")]
                    directory_length = len(audio_files)
                    if ratio < 1.0:
                        directory_length = int(directory_length * ratio)
                    
                    # Collect samples (audio path and corresponding transcript)
                    for audio_file in audio_files[:directory_length]:
                        audio_path = os.path.join(chapter_dir, audio_file)
                        transcript = transcripts.get(audio_file.split(".")[0])
                        if transcript:
                            self.samples.append((audio_path, transcript))

    def __len__(self):
        return len(self.samples)

    def add_noise(self, waveform):
        """
        Injects noise into the waveform based on the specified noise type and intensity.
        """
        if self.noise_type == "gaussian":
            noise = torch.randn_like(waveform) * self.noise_intensity
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(waveform) - 0.5) * 2 * self.noise_intensity
        elif self.noise_type == "adversarial":
            # Placeholder for adversarial noise generation (FGSM, PGD, etc.)
            noise = torch.zeros_like(waveform)  # Replace with adversarial noise logic
        else:
            noise = torch.zeros_like(waveform)  # No noise
        return waveform + noise

    def __getitem__(self, idx):
        audio_path, transcript = self.samples[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Optional transformations (e.g., resampling or augmentations)
        if sample_rate != 16000:  # Ensure consistent sample rate
            resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        # Apply noise if specified
        waveform = self.add_noise(waveform)
        
        return waveform, sample_rate, transcript


def custom_collate_fn(batch):
    """
    Handles variable-length waveforms by padding them to the same length.
    """
    waveforms, sample_rates, transcripts = zip(*batch)
    
    # Find the maximum length of waveforms in the batch
    max_len = max(waveform.size(1) for waveform in waveforms)
    
    # Create a padded tensor of shape [batch_size, 1, max_len]
    padded_waveforms = torch.zeros(len(waveforms), 1, max_len)
    
    # Copy each waveform into the padded tensor
    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, 0, :waveform.size(1)] = waveform

    return padded_waveforms, sample_rates, transcripts


# Example usage
if __name__ == "__main__":
    # Root directories for the LibriSpeech dataset
    root_dirs = [
        "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/train-clean-100",
        "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/train-clean-360",
        "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/train-other-500",
        "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/dev-clean",
        "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/dev-other",
        "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/test-clean",
        "/home/biometrics/hrishikesh/idl_project/LibriSpeech/LibriSpeech/test-other"
    ]

    # Initialize dataset with noise injection
    dataset = LibriSpeechDatasetWithNoise(
        root_dirs=root_dirs,
        noise_type="gaussian",  # Options: "gaussian", "uniform", "adversarial", None
        noise_intensity=0.01
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    # Test the dataloader
    for batch_idx, (padded_waveforms, sample_rates, transcripts) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f" - Padded waveforms shape: {padded_waveforms.shape}")
        print(f" - Sample rates: {sample_rates[:2]}")  # Print first two sample rates
        print(f" - Transcripts: {transcripts[:2]}")  # Print first two transcripts
        if batch_idx == 1:  # Only check the first 2 batches
            break