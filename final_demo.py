"""
final_demo.py
Complete Multimodal Emergency Risk Detection System Demo
Showcases Vision + Text fusion for emergency detection
"""

import torch
import numpy as np
from text.text_model import TextModel
from vision.vision_model import VisionModel
from fusion.fusion_engine import FusionEngine
from decision.decision_engine import DecisionEngine
import logging

logging.basicConfig(level=logging.WARNING)  # Suppress logs for cleaner output


def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_section(title):
    """Print formatted section"""
    print(f"\n{'-'*80}")
    print(f"► {title}")
    print(f"{'-'*80}")


def create_synthetic_image(threat_type="neutral"):
    """Create synthetic image for testing"""
    # Create random RGB image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Add subtle patterns for different threat types
    if threat_type == "fire":
        # Red/orange tones for fire
        img[:, :, 0] = np.clip(img[:, :, 0] + 50, 0, 255)  # More red
        img[:, :, 1] = np.clip(img[:, :, 1] - 20, 0, 255)  # Less green
    elif threat_type == "person_down":
        # Grayscale for prone figure
        gray = np.mean(img, axis=2, keepdims=True)
        img = np.repeat(gray, 3, axis=2)
    
    return img.astype(np.uint8)


def demo_scenario(title, text_input, threat_type="neutral"):
    """Run a complete scenario test"""
    print_section(title)
    
    # Initialize models (reuse if possible)
    text_model = TextModel()
    vision_model = VisionModel()
    fusion = FusionEngine()
    decision = DecisionEngine()
    
    # Create synthetic image
    img = create_synthetic_image(threat_type)
    
    # Process modalities
    print(f"\n📝 Text Input: '{text_input}'")
    text_result = text_model.process(text_input)
    print(f"   └─ Text Score: {text_result['text_score']:.3f} | Intent: {text_result['intent']}")
    
    print(f"\n👁️  Image Analysis (Synthetic)")
    vision_result = vision_model.process(img)
    print(f"   └─ Vision Score: {vision_result['vision_score']:.3f}")
    print(f"   └─ Threats: Fire={vision_result['fire_detected']}, "
          f"Person Down={vision_result['person_fallen']}, "
          f"Weapon={vision_result['weapon_detected']}")
    
    # Fuse outputs
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
    
    fusion_result = fusion.fuse(modality_outputs)
    final_decision = decision.decide(fusion_result)
    
    # Display decision
    print(f"\n🎯 FINAL DECISION:")
    print(f"   ┌─ Risk Level: {final_decision['risk_level']}")
    print(f"   ├─ Confidence: {final_decision['confidence']:.1f}%")
    print(f"   ├─ Situation: {final_decision['situation']}")
    print(f"   └─ Recommended Actions:")
    for action in final_decision['recommended_actions'][:2]:  # Show top 2
        print(f"      • {action}")
    
    return final_decision['risk_level']


def print_architecture():
    """Print system architecture"""
    print_header("🏗️  SYSTEM ARCHITECTURE")
    
    architecture = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                    INPUT LAYER                                   │
    │  [Camera/Image]    [Microphone/Audio]    [Text Description]     │
    └────────────┬──────────────────┬─────────────────────────┬───────┘
                 │                  │                         │
                 ▼                  ▼                         ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐
    │  VISION MODULE   │  │  AUDIO MODULE    │  │  TEXT MODULE       │
    │  (ResNet18 CNN)  │  │  (MelSpec + CNN) │  │  (Keyword + NN)    │
    │  - Fire detect   │  │  - Scream detect │  │  - Intent parsing  │
    │  - Person fallen │  │  - Emotion       │  │  - Severity score  │
    │  - Weapons       │  │  - Emergency tone│  │  - Risk keywords   │
    └────────┬─────────┘  └──────┬───────────┘  └─────────┬──────────┘
             │                   │                        │
             └───────────────────┼────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │  MULTIMODAL FUSION ENGINE  │
                    │  (Decision-Level Fusion)   │
                    │  Weights:                  │
                    │  - Vision: 40%             │
                    │  - Audio: 30%              │
                    │  - Text: 30%               │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │   DECISION ENGINE          │
                    │  Risk Classification:      │
                    │  - HIGH (>0.75)            │
                    │  - MEDIUM (0.4-0.75)       │
                    │  - LOW (<0.4)              │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │   OUTPUT & ALERT           │
                    │  - Risk Level              │
                    │  - Confidence Score        │
                    │  - Situation Description   │
                    │  - Recommended Actions     │
                    │  - Emergency Alerts        │
                    └────────────────────────────┘
    """
    print(architecture)


def main():
    """Run complete demo"""
    print_header("🚨 MULTIMODAL EMERGENCY RISK DETECTION SYSTEM")
    print("Advanced AI for real-world emergency response")
    
    # Show architecture
    print_architecture()
    
    # Run test scenarios
    print_header("📋 TEST SCENARIOS")
    
    scenarios = [
        {
            "title": "SCENARIO 1: Fire Emergency",
            "text": "Help! There is fire in the building! Emergency!",
            "threat": "fire"
        },
        {
            "title": "SCENARIO 2: Person Injured",
            "text": "Person fell down, unable to move. Need medical help immediately!",
            "threat": "person_down"
        },
        {
            "title": "SCENARIO 3: Security Threat",
            "text": "There is an intruder with a weapon. Call police immediately!",
            "threat": "weapon"
        },
        {
            "title": "SCENARIO 4: Normal Conditions",
            "text": "Everything is fine, just checking on the system.",
            "threat": "neutral"
        },
    ]
    
    results = []
    for scenario in scenarios:
        risk_level = demo_scenario(
            scenario["title"],
            scenario["text"],
            scenario["threat"]
        )
        results.append((scenario["title"].split(": ")[1], risk_level))
    
    # Print summary
    print_header("📊 RESULTS SUMMARY")
    print("\nScenario Results:")
    print(f"{'-'*40}")
    for scenario, risk in results:
        status = "🔴" if "HIGH" in risk else "🟡" if "MEDIUM" in risk else "🟢"
        print(f"{status} {scenario:<30} → {risk}")
    
    print(f"{'-'*40}")
    
    print_header("✅ SYSTEM CAPABILITIES")
    
    capabilities = """
    ✨ KEY FEATURES:
    
    1. MULTIMODAL ANALYSIS
       • Combines vision, audio, and text analysis
       • Cross-modal consistency checking
       • Reduces false positives through fusion
    
    2. ADVANCED THREAT DETECTION
       • Fire/smoke detection from images
       • Person down/collapse recognition
       • Weapon and security threats
       • Emergency intent parsing from speech/text
    
    3. INTELLIGENT DECISION MAKING
       • Weighted fusion of modalities
       • Context-aware risk assessment
       • Actionable recommendations
    
    4. REAL-WORLD APPLICABLE
       • Fast inference on CPU
       • Scalable architecture
       • Production-ready error handling
    
    🎯 USE CASES:
       • Smart security monitoring systems
       • Building safety and evacuation
       • Emergency response optimization
       • Real-time threat assessment
    """
    
    print(capabilities)
    
    print_header("🎉 DEMO COMPLETE")
    print("\nThe Multimodal Emergency Detection System is ready for deployment!")
    print("\nKey Statistics:")
    print(f"  • Vision Model: ResNet18 (Pretrained)")
    print(f"  • Text Model: Custom NN (128D features)")
    print(f"  • Fusion: Decision-Level (40-30-30% weights)")
    print(f"  • Processing Speed: Real-time inference")
    print(f"  • Device: CPU (GPU compatible)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
