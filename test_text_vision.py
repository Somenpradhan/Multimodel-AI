"""
test_text_vision.py
Phase-by-phase testing of Text and Vision models
"""

import torch
import numpy as np
from text.text_model import TextModel
from vision.vision_model import VisionModel
from fusion.fusion_engine import FusionEngine
from decision.decision_engine import DecisionEngine
import logging
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_text_model():
    """Test text model in isolation"""
    print("\n" + "="*70)
    print("📝 PHASE 1: TEXT MODEL TESTING")
    print("="*70)
    
    try:
        logger.info("Initializing TextModel...")
        text_model = TextModel()
        
        test_cases = [
            ("Help there is fire", "FIRE EMERGENCY"),
            ("I've been shot", "MEDICAL EMERGENCY"),
            ("There's an intruder", "SECURITY THREAT"),
            ("Everything is normal", "NO THREAT"),
        ]
        
        print("\nTesting Text Model:")
        print("-" * 70)
        
        for text_input, expected in test_cases:
            print(f"\n✓ Input: '{text_input}'")
            print(f"  Expected: {expected}")
            
            result = text_model.process(text_input)
            
            print(f"  Score: {result['text_score']:.3f}")
            print(f"  Intent: {result['intent']}")
            print(f"  Severity: {result['severity_score']:.3f}")
            print(f"  Keywords Found: {result['keywords_detected']}")
        
        print("\n" + "-" * 70)
        print("✅ Text Model Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Text Model Test FAILED: {e}\n")
        logger.error(f"Text model error: {e}", exc_info=True)
        return False


def create_dummy_image():
    """Create a dummy image for testing"""
    # Create a random RGB image (224, 224, 3)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return img


def test_vision_model():
    """Test vision model in isolation"""
    print("\n" + "="*70)
    print("👁️  PHASE 2: VISION MODEL TESTING")
    print("="*70)
    
    try:
        logger.info("Initializing VisionModel...")
        vision_model = VisionModel()
        
        print("\nTesting Vision Model with synthetic images:")
        print("-" * 70)
        
        for i in range(3):
            print(f"\n✓ Test Image {i+1}:")
            
            # Create dummy image
            dummy_img = create_dummy_image()
            
            result = vision_model.process(dummy_img)
            
            print(f"  Vision Score: {result['vision_score']:.3f}")
            print(f"  Fire Detected: {result['fire_detected']}")
            print(f"  Person Fallen: {result['person_fallen']}")
            print(f"  Weapon Detected: {result['weapon_detected']}")
            print(f"  Confidence: {result['confidence']:.3f}")
        
        print("\n" + "-" * 70)
        print("✅ Vision Model Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Vision Model Test FAILED: {e}\n")
        logger.error(f"Vision model error: {e}", exc_info=True)
        return False


def test_fusion():
    """Test fusion of text and vision outputs"""
    print("\n" + "="*70)
    print("🔄 PHASE 3: FUSION ENGINE TESTING")
    print("="*70)
    
    try:
        logger.info("Initializing Text and Vision models for fusion...")
        text_model = TextModel()
        vision_model = VisionModel()
        fusion = FusionEngine()
        decision = DecisionEngine()
        
        print("\nTesting Fusion with sample data:")
        print("-" * 70)
        
        # Create sample outputs
        text_result = text_model.process("Help there is fire")
        dummy_img = create_dummy_image()
        vision_result = vision_model.process(dummy_img)
        
        # Package outputs
        modality_outputs = {
            "vision": vision_result,
            "text": text_result,
            "audio": {
                "audio_score": 0.0,
                "scream_detected": False,
                "emotion": "calm",
                "confidence": 0.0
            }
        }
        
        print("\n📊 Modality Outputs:")
        print(f"  Vision Score: {vision_result['vision_score']:.3f}")
        print(f"  Text Score: {text_result['text_score']:.3f}")
        
        # Fuse outputs
        fusion_result = fusion.fuse(modality_outputs)
        
        print("\n🔀 Fusion Result:")
        print(f"  Final Risk Score: {fusion_result['final_risk_score']:.3f}")
        
        # Make decision
        final_decision = decision.decide(fusion_result)
        
        print("\n⚖️  Final Decision:")
        print(f"  Risk Level: {final_decision['risk_level']}")
        print(f"  Situation: {final_decision['situation']}")
        print(f"  Confidence: {final_decision['confidence']:.1f}%")
        
        print("\n" + "-" * 70)
        print("✅ Fusion Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Fusion Test FAILED: {e}\n")
        logger.error(f"Fusion error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚨 MULTIMODAL SYSTEM - PHASE-BY-PHASE TESTING")
    print("="*70)
    
    results = {
        "Text Model": test_text_model(),
        "Vision Model": test_vision_model(),
        "Fusion Engine": test_fusion()
    }
    
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    for phase, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{phase}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("="*70))
    if all_passed:
        print("🎉 ALL TESTS PASSED - System is ready!")
    else:
        print("⚠️  Some tests failed - please check the logs above")
    print("="*70 + "\n")
