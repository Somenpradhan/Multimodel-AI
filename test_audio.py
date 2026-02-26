from audio.audio_model import AudioModel

if __name__ == '__main__':
    am = AudioModel()
    print('Model initialized')
    result = am.process('audio1.mpeg')
    print('Result:', result)
