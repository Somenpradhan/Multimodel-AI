"""
audio_preprocess.py
Audio preprocessing module using Librosa
"""

import librosa
import numpy as np
import torch
import config
import logging


class AudioPreprocessor:
    def __init__(self):
        self.sr = config.SAMPLE_RATE
        self.n_mels = config.N_MELS
        self.hop_length = config.HOP_LENGTH
        self.n_fft = config.N_FFT
        self.duration = config.AUDIO_DURATION
        self.device = config.DEVICE

    def load_audio(self, audio_path):
        """
        Load audio file
        Returns: ndarray (samples,)
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            return y
        except Exception as e:
            logging.error(f"[AudioPreprocessor] Failed to load audio: {e}")
            raise

    def extract_mel_spectrogram(self, y):
        """
        Convert audio to Mel Spectrogram
        Returns: ndarray (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db

    def extract_mfcc(self, y):
        """
        Extract MFCC features for emotion detection
        """
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])

    def preprocess(self, audio_path):
        """
        Full audio preprocessing pipeline
        Returns: tensor (1, n_mels, time_steps)
        """
        try:
            # Load audio
            y = self.load_audio(audio_path)
            
            # Extract mel spectrogram
            mel_spec = self.extract_mel_spectrogram(y)
            
            # Pad/trim to fixed size
            target_length = int(self.sr * self.duration / self.hop_length)
            if mel_spec.shape[1] < target_length:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, target_length - mel_spec.shape[1])))
            else:
                mel_spec = mel_spec[:, :target_length]
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            
            return audio_tensor
        
        except Exception as e:
            logging.error(f"[AudioPreprocessor] Preprocessing failed: {e}")
            raise
