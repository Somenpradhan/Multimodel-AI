import config
import logging


class DecisionEngine:
    """
    Decision Engine
    Converts risk scores to actionable decisions and recommendations
    """

    def __init__(self):
        self.high_risk_threshold = config.HIGH_RISK_THRESHOLD
        self.medium_risk_threshold = config.MEDIUM_RISK_THRESHOLD
        
        self.recommendations = {
            "HIGH RISK": [
                "🚨 IMMEDIATE EVACUATION REQUIRED",
                "📞 Call emergency services (911/Police)",
                "🚪 Exit to nearest safe location",
                "🚨 Alert others nearby",
                "📹 Document evidence if safe to do so"
            ],
            "MEDIUM RISK": [
                "⚠️  Prepare to evacuate",
                "📱 Keep phone accessible",
                "🚪 Note exits and safe locations",
                "👥 Alert nearby individuals",
                "⏱️  Monitor situation closely"
            ],
            "LOW RISK": [
                "✅ Monitor situation",
                "🔍 Stay aware of surroundings",
                "📱 Have phone ready",
                "👀 Report any changes to authorities",
                "✔️  Proceed with caution"
            ]
        }

    def decide(self, fusion_result):
        """
        Convert risk score to decision with detailed recommendations
        
        Args:
            fusion_result: dict with 'final_risk_score' and modality details
        
        Returns:
            dict with risk level, confidence, situation, and recommendations
        """
        try:
            if isinstance(fusion_result, dict):
                risk_score = fusion_result.get("final_risk_score", 0.0)
                modality_details = fusion_result.get("modality_details", {})
            else:
                # Backward compatibility - if just a float is passed
                risk_score = fusion_result
                modality_details = {}
            
            # Classify risk level
            if risk_score > self.high_risk_threshold:
                risk_level = "HIGH RISK"
                risk_class = 2
            elif risk_score > self.medium_risk_threshold:
                risk_level = "MEDIUM RISK"
                risk_class = 1
            else:
                risk_level = "LOW RISK"
                risk_class = 0
            
            # Determine situation from modalities
            situation = self._determine_situation(modality_details)
            
            # Get recommendations
            actions = self.recommendations.get(risk_level, [])
            
            result = {
                "risk_level": risk_level,
                "risk_class": risk_class,
                "confidence": min(risk_score * 100, 100.0),
                "risk_score": risk_score,
                "situation": situation,
                "recommended_actions": actions,
                "modality_breakdown": {
                    "vision": modality_details.get("vision", {}),
                    "audio": modality_details.get("audio", {}),
                    "text": modality_details.get("text", {})
                }
            }
            
            logging.info(f"[DecisionEngine] Decision: {risk_level} (Score: {risk_score:.3f})")
            logging.info(f"[DecisionEngine] Situation: {situation}")
            
            return result
        
        except Exception as e:
            logging.error(f"[DecisionEngine] Decision failed: {e}")
            return {
                "risk_level": "UNKNOWN",
                "risk_class": -1,
                "confidence": 0.0,
                "risk_score": 0.0,
                "situation": "Error in risk assessment",
                "recommended_actions": ["⚠️  Manual verification required"],
                "error": str(e)
            }

    def _determine_situation(self, modality_details):
        """
        Determine the emergency situation from modality details
        """
        vision_data = modality_details.get("vision", {})
        audio_data = modality_details.get("audio", {})
        text_data = modality_details.get("text", {})
        
        # Collect detected hazards
        hazards = []
        
        if vision_data.get("fire_detected"):
            hazards.append("Fire/Smoke")
        if vision_data.get("weapon_detected"):
            hazards.append("Weapon")
        if vision_data.get("person_fallen"):
            hazards.append("Person Down")
        
        if audio_data.get("scream_detected"):
            hazards.append("Scream Detected")
        if audio_data.get("emotion") == "panic":
            hazards.append("Panic Signals")
        
        intent = text_data.get("intent", "")
        if intent:
            hazards.append(intent.replace("_", " ").title())
        
        if hazards:
            return " + ".join(hazards)
        else:
            return "Unknown situation"