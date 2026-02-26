"""
demo.py
Demonstration of the Multimodal Emergency Detection System
Uses synthetic data to showcase the system without requiring actual media files
"""

import numpy as np
import torch
from vision.vision_model import VisionModel
from audio.audio_model import AudioModel
from text.text_model import TextModel
from fusion.fusion_engine import FusionEngine
from decision.decision_engine import DecisionEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_vision_output():
    """Simulate vision model output"""
    return {
        "vision_score": 0.82,
        "fire_detected": True,
        "person_fallen": False,
        "weapon_detected": False,
        "confidence": 0.82
    }


def create_dummy_audio_output():
    """Simulate audio model output"""
    return {
        "audio_score": 0.75,
        "scream_detected": True,
        "emotion": "panic",
        "confidence": 0.75
    }


def create_dummy_text_output():
    """Simulate text model output"""
    return {
        "text_score": 0.91,
        "intent": "fire_emergency",
        "severity_score": 0.91,
        "confidence": 0.91,
        "keywords_detected": True
    }


def demo_real_models():
    """Demo using real text model with sample inputs"""
    print("\n" + "="*70)
    print("🚨 MULTIMODAL EMERGENCY DETECTION SYSTEM - REAL MODEL DEMO 🚨")
    print("="*70)
    
    logger.info("Initializing models for real demo...")
    
    # Initialize models
    vision_model = VisionModel()
    audio_model = AudioModel()
    text_model = TextModel()
    fusion = FusionEngine()
    decision = DecisionEngine()
    
    logger.info("Models initialized")
    
    # Simulate modality outputs
    outputs = {
        "vision": create_dummy_vision_output(),
        "audio": create_dummy_audio_output(),
        "text": text_model.process("Help there is fire! Emergency!")
    }
    
    logger.info("Modality outputs generated")
    
    # Print modality outputs
    print("\n📊 MODALITY OUTPUTS:")
    print("-" * 70)
    print(f"Vision Score: {outputs['vision']['vision_score']:.3f}")
    print(f"  └─ Fire Detected: {outputs['vision']['fire_detected']}")
    print(f"Audio Score: {outputs['audio']['audio_score']:.3f}")
    print(f"  └─ Emotion: {outputs['audio']['emotion']}")
    print(f"Text Score: {outputs['text']['text_score']:.3f}")
    print(f"  └─ Intent: {outputs['text']['intent']}")
    
    # Fuse outputs
    print("\n🔄 FUSION PROCESS:")
    print("-" * 70)
    fusion_result = fusion.fuse(outputs)
    print(f"Fused Risk Score: {fusion_result['final_risk_score']:.3f}")
    
    # Make decision
    print("\n⚖️  DECISION:")
    print("-" * 70)
    final_decision = decision.decide(fusion_result)
    
    print(f"Risk Level: {final_decision['risk_level']}")
    print(f"Confidence: {final_decision['confidence']:.1f}%")
    print(f"Situation: {final_decision['situation']}")
    
    print("\n📋 RECOMMENDED ACTIONS:")
    print("-" * 70)
    for i, action in enumerate(final_decision['recommended_actions'], 1):
        print(f"{i}. {action}")
    
    print("\n" + "="*70 + "\n")
    
    return final_decision


def demo_text_only():
    """Demo text model with various inputs"""
    print("\n" + "="*70)
    print("📝 TEXT MODEL - INTENT & SEVERITY CLASSIFICATION DEMO")
    print("="*70)
    
    text_model = TextModel()
    
    test_cases = [
        "Help there is fire",
        "I've been shot, please call 911",
        "There's an intruder with a weapon",
        "Everything is fine",
        "Someone fell and isn't moving",
    ]
    
    for text in test_cases:
        print(f"\nInput: '{text}'")
        result = text_model.process(text)
        print(f"  Score: {result['text_score']:.3f}")
        print(f"  Intent: {result['intent']}")
        print(f"  Severity: {result['severity_score']:.3f}")


if __name__ == "__main__":
    print("\n🎯 RUNNING MULTIMODAL SYSTEM DEMO\n")
    
    # Run text-only demo first (doesn't need file I/O)
    demo_text_only()
    
    print("\n" + "="*70)
    print("Ready to process real inputs when image/audio files are available.")
    print("Update 'main.py' with your image and audio file paths.")
    print("="*70 + "\n")
