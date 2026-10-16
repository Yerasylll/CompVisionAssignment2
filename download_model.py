from pathlib import Path
from urllib.request import urlretrieve

URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
path = Path(__file__).parent / 'models' / 'gesture_recognizer.task'
path.parent.mkdir(exist_ok=True)
if not path.exists() or path.stat().st_size < 100000:
    print('Downloading official MediaPipe Gesture Recognizer...')
    urlretrieve(URL, path)
print(f'Model ready: {path} ({path.stat().st_size / 1e6:.1f} MB)')
