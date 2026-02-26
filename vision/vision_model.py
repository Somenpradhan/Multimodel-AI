from vision.vision_preprocess import VisionPreprocessor
import torch
import torch.nn as nn
import torchvision.models as models
import config
import logging


class SimpleCNN(nn.Module):
    """Lightweight CNN for emergency detection"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Pre-trained ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Remove last classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class VisionModel:

    def __init__(self, model=None):
        self.preprocessor = VisionPreprocessor()
        self.device = config.DEVICE
        
        # Use provided model or create default
        if model is None:
            self.model = SimpleCNN().to(self.device)
            self.model.eval()
        else:
            self.model = model.to(self.device)
            self.model.eval()
        
        logging.info(f"[VisionModel] Initialized on device: {self.device}")

    def process(self, image_input):
        """
        Process image and detect emergency indicators
        Fire, weapons, falls, etc.
        """
        try:
            # Preprocess image
            image_tensor = self.preprocessor.preprocess(image_input)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(image_tensor)
            
            score = float(output.item())
            
            return {
                "vision_score": score,
                "fire_detected": score > 0.6,
                "person_fallen": score > 0.5,
                "weapon_detected": score > 0.7,
                "confidence": score
            }
        
        except Exception as e:
            logging.error(f"[VisionModel] Processing failed: {e}")
            return {
                "vision_score": 0.0,
                "fire_detected": False,
                "person_fallen": False,
                "weapon_detected": False,
                "confidence": 0.0,
                "error": str(e)
            }