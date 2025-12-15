import sys
import types
import importlib
from types import SimpleNamespace
import builtins
import pytest


# Helper: install a fake cv2 module into sys.modules to avoid real OpenCV dependency
class FakeCV2ModuleFactory:
    def __init__(self):
        self.reset()

    def build(self):
        mod = types.ModuleType('cv2')
        # constants
        mod.FONT_HERSHEY_SIMPLEX = 0
        mod.LINE_AA = 16
        mod.COLOR_BGR2GRAY = 6
        mod.CAP_PROP_FRAME_WIDTH = 3
        mod.CAP_PROP_FRAME_HEIGHT = 4

        # state holders
        mod._cascade_empty = self._cascade_empty
        mod._detect_faces = self._detect_faces
        mod._imread_return = self._imread_return
        mod._imwrite_calls = []
        mod._rectangle_calls = []
        mod._put_text_calls = []
        mod._imshow_calls = []
        mod._destroy_all_called = False
        mod._waitKey_return = self._waitKey_return

        # video state
        mod._video_open = self._video_open
        mod._frames = list(self._frames)
        mod._frame_size = self._frame_size

        # writer state
        mod._writer_open = self._writer_open
        mod._writer_writes = []

        class CascadeClassifier:
            def __init__(self, path):
                self.path = path

            def empty(self):
                return mod._cascade_empty

            def detectMultiScale(self, gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
                return list(mod._detect_faces)

        class VideoCapture:
            def __init__(self, source):
                self.source = source
                self._open = mod._video_open
                self._frames = list(mod._frames)
                self._frame_size = mod._frame_size

            def isOpened(self):
                return self._open

            def read(self):
                if self._frames:
                    return True, self._frames.pop(0)
                return False, None

            def get(self, prop):
                if prop == mod.CAP_PROP_FRAME_WIDTH:
                    return self._frame_size[0]
                if prop == mod.CAP_PROP_FRAME_HEIGHT:
                    return self._frame_size[1]
                return 0

            def release(self):
                self._open = False

        class VideoWriter:
            def __init__(self, path, fourcc, fps, size):
                self.path = path
                self.fourcc = fourcc
                self.fps = fps
                self.size = size
                self._open = mod._writer_open

            def isOpened(self):
                return self._open

            def write(self, frame):
                mod._writer_writes.append(frame)

            def release(self):
                self._open = False

        # module-level API
        mod.CascadeClassifier = CascadeClassifier
        mod.VideoCapture = VideoCapture
        mod.VideoWriter = VideoWriter

        def VideoWriter_fourcc(a, b, c, d):
            return 0

        def imread(path):
            return mod._imread_return

        def imwrite(path, img):
            mod._imwrite_calls.append((path, img))
            return True

        def cvtColor(img, code):
            return img

        def rectangle(img, pt1, pt2, color, thickness):
            mod._rectangle_calls.append((pt1, pt2, color, thickness))

        def putText(img, text, org, fontFace, fontScale, color, thickness, lineType):
            mod._put_text_calls.append((text, org))

        def imshow(winname, img):
            mod._imshow_calls.append(winname)

        def waitKey(delay):
            return mod._waitKey_return

        def destroyAllWindows():
            mod._destroy_all_called = True

        # bind functions
        mod.VideoWriter_fourcc = VideoWriter_fourcc
        mod.imread = imread
        mod.imwrite = imwrite
        mod.cvtColor = cvtColor
        mod.rectangle = rectangle
        mod.putText = putText
        mod.imshow = imshow
        mod.waitKey = waitKey
        mod.destroyAllWindows = destroyAllWindows
        return mod

    def reset(self):
        # defaults for a neutral fake environment
        self._cascade_empty = False
        self._detect_faces = []
        self._imread_return = object()  # any non-None object to represent an image
        self._waitKey_return = -1
        self._video_open = True
        self._frames = ['frame1']
        self._frame_size = (640, 480)
        self._writer_open = True


@pytest.fixture
def fake_cv2(monkeypatch):
    factory = FakeCV2ModuleFactory()
    mod = factory.build()
    # Install into sys.modules so that importing FaceDetection resolves cv2 to our fake
    monkeypatch.setitem(sys.modules, 'cv2', mod)
    yield mod


@pytest.fixture
def fd(fake_cv2, monkeypatch):
    # Ensure project root is importable and a fresh import of FaceDetection with the fake cv2 available
    from pathlib import Path
    proj_root = str(Path(__file__).resolve().parents[1])
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    if 'FaceDetection' in sys.modules:
        del sys.modules['FaceDetection']
    import FaceDetection as fd
    return fd


# Behaviors to test:
# 1) Should raise FileNotFoundError when cascade path does not exist
# 2) Should raise RuntimeError when cascade fails to load (empty)
# 3) Should process image input, draw detections, and save annotated image
# 4) Should error when video/camera source cannot be opened
# 5) Should continue without saving when video writer cannot open, logging a warning
# 6) Should overlay FPS text on video frames
# 7) Should handle display flag by opening window and closing on 'q' key for video
# 8) resolve_source should parse numeric camera index and file paths correctly


def make_args(**overrides):
    defaults = dict(
        source='0',
        cascade='haarcascade_frontalface_default.xml',
        scale_factor=1.1,
        min_neighbors=5,
        min_size=(30, 30),
        save_video='',
        output_fps=30.0,
        display=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_missing_cascade_raises(fd, fake_cv2, monkeypatch):
    # mock os.path.exists to return False
    monkeypatch.setattr(fd.os.path, 'exists', lambda p: False)
    monkeypatch.setattr(fd, 'parse_args', lambda: make_args())

    with pytest.raises(FileNotFoundError):
        fd.main()


def test_cascade_empty_raises(fd, fake_cv2, monkeypatch):
    # cascade file exists but classifier is empty
    fake_cv2._cascade_empty = True
    # Rebuild module to apply new state flags
    monkeypatch.setitem(sys.modules, 'cv2', fake_cv2)

    monkeypatch.setattr(fd.os.path, 'exists', lambda p: True)
    monkeypatch.setattr(fd, 'parse_args', lambda: make_args(source='0'))

    with pytest.raises(RuntimeError, match='Failed to load cascade'):
        fd.main()


def test_image_processing_saves_output(fd, fake_cv2, monkeypatch, capsys, tmp_path):
    # Arrange image path
    img_path = str(tmp_path / 'test.jpg')
    # For image branch, isfile must be True and exists True
    monkeypatch.setattr(fd.os.path, 'exists', lambda p: True)
    monkeypatch.setattr(fd.os.path, 'isfile', lambda p: p == img_path)

    # Fake image and two detected faces
    fake_cv2._imread_return = object()
    fake_cv2._detect_faces = [(10, 10, 20, 20), (30, 30, 15, 15)]

    monkeypatch.setattr(fd, 'parse_args', lambda: make_args(source=img_path, display=False))

    fd.main()

    out = capsys.readouterr().out
    assert 'Saved image with detections' in out
    # Should have saved output with _faces.png suffix
    assert any(call[0].endswith('_faces.png') for call in fake_cv2._imwrite_calls)
    # Rectangles drawn for each detection
    assert len(fake_cv2._rectangle_calls) == 2


def test_video_source_open_failure(fd, fake_cv2, monkeypatch):
    # Video capture fails to open
    fake_cv2._video_open = False
    monkeypatch.setitem(sys.modules, 'cv2', fake_cv2)

    monkeypatch.setattr(fd.os.path, 'exists', lambda p: True)
    monkeypatch.setattr(fd, 'parse_args', lambda: make_args(source='0'))

    with pytest.raises(RuntimeError, match='Could not open video source'):
        fd.main()


def test_video_writer_fallback_warning(fd, fake_cv2, monkeypatch, capsys):
    # Video opens with one frame; writer fails to open; loop ends after one frame
    fake_cv2._video_open = True
    fake_cv2._frames = ['frame1']
    fake_cv2._writer_open = False
    monkeypatch.setitem(sys.modules, 'cv2', fake_cv2)

    monkeypatch.setattr(fd.os.path, 'exists', lambda p: True)
    monkeypatch.setattr(fd, 'parse_args', lambda: make_args(source='0', save_video='out.mp4', display=False))

    fd.main()
    out = capsys.readouterr().out
    assert 'could not open video writer' in out.lower()


def test_draw_fps_overlays_text(fd, fake_cv2):
    frame = object()
    fd.draw_fps(frame, 12.34)
    assert any(t[0].startswith('FPS:') for t in fake_cv2._put_text_calls)


def test_video_display_exits_on_q(fd, fake_cv2, monkeypatch):
    # Configure one frame and simulate pressing 'q'
    fake_cv2._video_open = True
    fake_cv2._frames = ['frame1']
    fake_cv2._writer_open = False
    fake_cv2._waitKey_return = ord('q')
    monkeypatch.setitem(sys.modules, 'cv2', fake_cv2)

    monkeypatch.setattr(fd.os.path, 'exists', lambda p: True)
    monkeypatch.setattr(fd, 'parse_args', lambda: make_args(source='0', display=True))

    fd.main()
    assert fake_cv2._imshow_calls  # window should have been shown
    assert fake_cv2._destroy_all_called  # windows should be destroyed at the end


def test_resolve_source_parsing(fd):
    val, is_cam = fd.resolve_source('0')
    assert val == 0 and is_cam is True
    val, is_cam = fd.resolve_source('video.mp4')
    assert val == 'video.mp4' and is_cam is False
