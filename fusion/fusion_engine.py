import numpy as np
import config
import logging


class FusionEngine:
    """
    Multimodal Fusion Engine
    Combines scores from Vision, Audio, and Text modalities
    Uses weighted decision-level fusion
    """

    def __init__(self):
        self.vision_weight = config.VISION_WEIGHT
        self.audio_weight = config.AUDIO_WEIGHT
        self.text_weight = config.TEXT_WEIGHT
        
        logging.info("[FusionEngine] Initialized with weights:")
        logging.info(f"  Vision: {self.vision_weight}")
        logging.info(f"  Audio: {self.audio_weight}")
        logging.info(f"  Text: {self.text_weight}")

    def fuse(self, modality_outputs):
        """
        Fuse multimodal outputs using weighted combination
        
        Args:
            modality_outputs: dict with keys "vision", "audio", "text"
            Each contains scores and additional metadata
        
        Returns:
            float: Final risk score [0, 1]
        """
        try:
            # Extract scores from each modality
            vision_score = modality_outputs.get("vision", {}).get("vision_score", 0.0)
            audio_score = modality_outputs.get("audio", {}).get("audio_score", 0.0)
            text_score = modality_outputs.get("text", {}).get("text_score", 0.0)
            
            # Determine which modalities are available (non-zero)
            active = []
            if vision_score > 0.0:
                active.append("vision")
            if audio_score > 0.0:
                active.append("audio")
            if text_score > 0.0:
                active.append("text")
            
            # Renormalize weights if some modalities missing
            if len(active) < 3:
                # get original weights for active ones
                vw = self.vision_weight if "vision" in active else 0.0
                aw = self.audio_weight if "audio" in active else 0.0
                tw = self.text_weight if "text" in active else 0.0
                total = vw + aw + tw
                if total > 0:
                    vw /= total
                    aw /= total
                    tw /= total
                else:
                    # fallback equal share among present
                    n = max(1, len(active))
                    vw = aw = tw = 1.0 / n if len(active) > 0 else 0.0
            else:
                vw = self.vision_weight
                aw = self.audio_weight
                tw = self.text_weight
            
            # Weighted fusion with adaptive weights
            final_score = (
                vw * vision_score +
                aw * audio_score +
                tw * text_score
            )
            
            # Clamp to [0, 1]
            final_score = np.clip(final_score, 0.0, 1.0)
            
            # Log fusion details
            logging.info(f"[FusionEngine] Fusion result: {final_score:.3f}")
            logging.info(f"  Active modalities: {active}")
            logging.info(f"  Vision: {vision_score:.3f} × {vw:.3f}")
            logging.info(f"  Audio: {audio_score:.3f} × {aw:.3f}")
            logging.info(f"  Text: {text_score:.3f} × {tw:.3f}")
            
            return {
                "final_risk_score": final_score,
                "vision_score": vision_score,
                "audio_score": audio_score,
                "text_score": text_score,
                "active_modalities": active,
                "modality_details": modality_outputs
            }
        
        except Exception as e:
            logging.error(f"[FusionEngine] Fusion failed: {e}")
            return {
                "final_risk_score": 0.0,
                "vision_score": 0.0,
                "audio_score": 0.0,
                "text_score": 0.0,
                "error": str(e)
            }

    def cross_modal_check(self, modality_outputs):
        """
        Cross-modal consistency check
        Penalizes conflicting signals to reduce false positives
        """
        vision_score = modality_outputs.get("vision", {}).get("vision_score", 0.0)
        audio_score = modality_outputs.get("audio", {}).get("audio_score", 0.0)
        text_score = modality_outputs.get("text", {}).get("text_score", 0.0)
        
        # If scores are very conflicting, reduce final confidence
        scores = [vision_score, audio_score, text_score]
        std_dev = np.std(scores)
        
        # High std_dev means conflicting signals
        if std_dev > 0.5:
            confidence_penalty = std_dev * 0.2
            logging.warning(f"[FusionEngine] Conflicting signals detected. Confidence penalty: {confidence_penalty:.3f}")
            return confidence_penalty
        
        return 0.0