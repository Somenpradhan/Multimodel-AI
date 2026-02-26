import torch
import torch.nn as nn
import numpy as np
import config
import logging


class TextClassifier(nn.Module):
    """Simple neural network for text danger classification"""
    def __init__(self, input_size=128):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


class TextModel:

    def __init__(self, model=None):
        self.device = config.DEVICE
        
        # Danger keywords for rule-based detection
        self.danger_keywords = {
            "fire": 0.8,
            "help": 0.6,
            "emergency": 0.9,
            "attack": 0.9,
            "hurt": 0.7,
            "injured": 0.8,
            "accident": 0.7,
            "blood": 0.8,
            "dying": 0.95,
            "dead": 0.9,
            "gun": 0.85,
            "weapon": 0.8,
            "bomb": 0.95,
            "trap": 0.7,
            "danger": 0.8,
            "scream": 0.7,
            "cry": 0.6,
            "alarm": 0.75,
            "911": 0.9,
            "call": 0.5,
            "pain": 0.7
        }
        
        # Initialize neural model
        if model is None:
            self.model = TextClassifier(input_size=128).to(self.device)
            self.model.eval()
        else:
            self.model = model.to(self.device)
            self.model.eval()
        
        logging.info(f"[TextModel] Initialized on device: {self.device}")

    def preprocess(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            text = str(text)
        return text.lower().strip()

    def keyword_score(self, text):
        """Rule-based danger score from keywords"""
        text_lower = text.lower()
        max_score = 0.0
        matched_keywords = []
        
        for keyword, score in self.danger_keywords.items():
            if keyword in text_lower:
                max_score = max(max_score, score)
                matched_keywords.append(keyword)
        
        return max_score, matched_keywords

    def detect_intent(self, text):
        """Detect emergency intent from text"""
        text_lower = text.lower()
        
        intents = {
            "fire_emergency": ["fire", "smoke", "burn", "flame"],
            "medical_emergency": ["hurt", "bleeding", "injured", "accident", "dying", "pain"],
            "security_threat": ["attack", "gun", "weapon", "bomb", "intruder", "shot"],
            "general_distress": ["help", "emergency", "911", "call", "scream", "cry"],
        }
        
        for intent, keywords in intents.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        
        return "general_alert"

    def text_to_vector(self, text):
        """Convert text to fixed-size feature vector using character and word analysis"""
        text_lower = text.lower()
        
        # Create 128-dimensional feature vector
        features = np.zeros(128, dtype=np.float32)
        
        # Features 0-19: Character presence (a-z + space, punctuation)
        char_set = set(text_lower)
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyz "):
            if i < 128 and char in char_set:
                features[i] = 1.0
        
        # Features 20-50: Word-level danger indicators
        words = text_lower.split()
        for i, keyword in enumerate(list(self.danger_keywords.keys())[:31]):
            if keyword in text_lower:
                features[20 + i] = self.danger_keywords[keyword]
        
        # Features 51-60: Text statistics
        features[51] = len(words) / 20.0  # Word count (normalized)
        features[52] = len(text_lower) / 200.0  # Character count (normalized)
        
        # Features 53-80: Intent encoding
        intent = self.detect_intent(text)
        intent_map = {
            "fire_emergency": [1, 0, 0, 0],
            "medical_emergency": [0, 1, 0, 0],
            "security_threat": [0, 0, 1, 0],
            "general_distress": [0, 0, 0, 1],
            "general_alert": [0.5, 0.5, 0.5, 0.5]
        }
        intent_vec = intent_map.get(intent, intent_map["general_alert"])
        features[53:57] = intent_vec
        
        # Features 57-127: Padding/additional features
        features[57:128] = np.random.normal(0, 0.1, 71)  # Small noise
        
        return features

    def process(self, text):
        """
        Process text and classify emergency severity
        """
        try:
            # Preprocess
            cleaned_text = self.preprocess(text)
            
            # Get keyword-based score
            keyword_score, _ = self.keyword_score(cleaned_text)
            
            # Detect intent
            intent = self.detect_intent(cleaned_text)
            
            # Convert text to feature vector
            text_vector = self.text_to_vector(cleaned_text)
            text_tensor = torch.from_numpy(text_vector).float().unsqueeze(0).to(self.device)
            
            # Forward pass through neural model
            with torch.no_grad():
                neural_score = self.model(text_tensor).item()
            
            # Combine keyword and neural scores
            final_score = 0.6 * keyword_score + 0.4 * neural_score
            final_score = float(np.clip(final_score, 0.0, 1.0))
            
            return {
                "text_score": final_score,
                "intent": intent,
                "severity_score": final_score,
                "confidence": final_score,
                "keywords_detected": keyword_score > 0.5
            }
        
        except Exception as e:
            logging.error(f"[TextModel] Processing failed: {e}")
            return {
                "text_score": 0.0,
                "intent": "unknown",
                "severity_score": 0.0,
                "confidence": 0.0,
                "keywords_detected": False,
                "error": str(e)
            }