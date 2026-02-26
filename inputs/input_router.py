class InputRouter:

    def __init__(self, vision_model, audio_model, text_model):
        self.vision_model = vision_model
        self.audio_model = audio_model
        self.text_model = text_model

    def route(self, image=None, audio=None, text=None):

        outputs = {}

        if image is not None:
            outputs["vision"] = self.vision_model.process(image)

        if audio is not None:
            outputs["audio"] = self.audio_model.process(audio)

        if text is not None:
            outputs["text"] = self.text_model.process(text)

        return outputs