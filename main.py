"""
main.py
Multimodal Emergency Risk Detection System
Main entry point
"""

import logging
import sys
from inputs.input_router import InputRouter
from fusion.fusion_engine import FusionEngine
from decision.decision_engine import DecisionEngine
from vision.vision_model import VisionModel
from audio.audio_model import AudioModel
from text.text_model import TextModel
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_results(final_decision):
    """Pretty print the final decision"""
    print("\n" + "="*60)
    print(" MULTIMODAL EMERGENCY RISK DETECTION SYSTEM ")
    print("="*60)
    
    print(f"\n Risk Level: {final_decision['risk_level']}")
    print(f" Confidence: {final_decision['confidence']:.1f}%")
    print(f" Situation: {final_decision['situation']}")
    
    print("\n Recommended Actions:")
    for action in final_decision['recommended_actions']:
        print(f"   {action}")
    
    print("\n Modality Breakdown:")
    print(f"   Vision Score: {final_decision['modality_breakdown']['vision'].get('confidence', 0):.3f}")
    print(f"   Audio Score: {final_decision['modality_breakdown']['audio'].get('confidence', 0):.3f}")
    print(f"   Text Score: {final_decision['modality_breakdown']['text'].get('confidence', 0):.3f}")
    
    print("\n" + "="*60 + "\n")


def main():
    try:
        logger.info("Initializing Multimodal Emergency Detection System...")
        
        # Initialize models
        vision_model = VisionModel(model=None)
        audio_model = AudioModel(model=None)
        text_model = TextModel(model=None)
        
        logger.info("Models initialized successfully")
        
        # Initialize pipeline components
        router = InputRouter(vision_model, audio_model, text_model)
        fusion = FusionEngine()
        decision = DecisionEngine()
        
        logger.info("Pipeline ready for inference")
        
        # Example inputs
        image_path = "img.jpg"
        audio_path = "audio1.mpeg"
        text_input = "Help there is fire"
        
        print(f"\n Processing inputs:")
        print(f"   Image: {image_path}")
        print(f"   Audio: {audio_path}")
        print(f"   Text: {text_input}")
        
        # Route inputs through modalities
        logger.info("Routing inputs through modalities...")
        outputs = router.route(image=image_path, audio=audio_path, text=text_input)
        
        # Fuse modalities
        logger.info("Fusing multimodal outputs...")
        fusion_result = fusion.fuse(outputs)
        
        # Make final decision
        logger.info("Making final decision...")
        final_decision = decision.decide(fusion_result)
        
        # Display results
        print_results(final_decision)
        
        return final_decision
    
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        print(f"❌ Error: {e}")
        print("\nPlease provide valid input files:")
        print("  - img.jpg (for vision)")
        print("  - audio1.mpeg (for audio)")
        return None
    
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        print(f"❌ System error: {e}")
        return None


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)