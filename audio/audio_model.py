from audio.audio_preprocess import AudioPreprocessor
import torch
import torch.nn as nn
import config
import logging


class AudioCNN(nn.Module):
    """CNN for audio emotion and distress detection"""
    def __init__(self, n_mels=128, hidden_dim=128):
        super(AudioCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(n_mels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # instead of MaxPool we will use AdaptiveAvgPool to fix length
        self.pool = nn.AdaptiveAvgPool1d(16)  # output length 16 regardless of input
        self.dropout = nn.Dropout(0.3)
        
        # now input to fc1 is 256 * 16
        self.fc1 = nn.Linear(256 * 16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, n_mels, time_steps)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.sigmoid(self.fc2(x))
        return x


class AudioModel:

    def __init__(self, model=None):
        self.preprocessor = AudioPreprocessor()
        self.device = config.DEVICE
        
        if model is None:
            self.model = AudioCNN(n_mels=config.N_MELS).to(self.device)
            self.model.eval()
        else:
            self.model = model.to(self.device)
            self.model.eval()
        
        logging.info(f"[AudioModel] Initialized on device: {self.device}")

    def process(self, audio):
        """
        Process audio and detect distress signals
        Scream, panic tone, fire alarm sounds
        """
        try:
            # Preprocess audio to mel spectrogram
            audio_tensor = self.preprocessor.preprocess(audio)
            
            # Forward pass
            with torch.no_grad():
                prediction = self.model(audio_tensor)
            
            score = float(prediction.item())
            
            # Determine emotion based on score
            if score > 0.7:
                emotion = "panic"
            elif score > 0.4:
                emotion = "stressed"
            else:
                emotion = "calm"
            
            return {
                "audio_score": score,
                "scream_detected": score > 0.7,
                "emotion": emotion,
                "confidence": score
            }
        
        except Exception as e:
            logging.error(f"[AudioModel] Processing failed: {e}")
            return {
                "audio_score": 0.0,
                "scream_detected": False,
                "emotion": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }