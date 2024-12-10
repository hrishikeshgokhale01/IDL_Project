Automatic Speech Recognition (ASR) systems have achieved remarkable progress with the development of advanced neural architectures, particularly Convolutional Neural Networks (CNNs). While
CNNs demonstrate strong performance by modeling local spectral patterns and achieving invariance to small frequency shifts, their reliance on weight-sharing mechanisms poses limitations in biological
plausibility and robustness to adversarial perturbations. In this work, we investigate the Conformer model—a state-of-the-art ASR architecture that combines convolutional networks and self-attention
mechanisms to effectively capture both local and global dependencies in speech data. To address the limitations of standard CNNs, we introduce locally connected layers (untied convolutional kernels)
within the Conformer architecture. These layers replace traditional tied kernels with independently tuned weights, potentially enhancing robustness to adversarial and non-adversarial noise. The model
processes log Mel-spectrogram representations of speech waveforms from the LibriSpeech dataset as input and outputs corresponding text transcriptions as sequences of tokens. We evaluate the modified
Conformer on word error rate (WER) and noise robustness, highlighting the potential of locally connected layers to address key challenges in ASR systems.
