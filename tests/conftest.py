"""
Shared pytest fixtures and import-time stubs for STAR SARA unit tests.

``star_sara_v3`` imports a lot of heavy / hardware-bound third-party packages
at module load time (whisper + torch, PySide6, sounddevice, pygame, edge_tts,
groq, ...). None of them are needed to exercise the pure application logic
(memory / tasks / notes, intent classification, audio math, JSON helpers,
wake-word matching), and installing them (or having audio/display hardware)
would make the unit suite slow and non-hermetic.

To keep the tests fast and runnable anywhere (including headless CI), we insert
lightweight stub modules into ``sys.modules`` *before* importing the app. NumPy
is intentionally left as the real package because ``AudioProcessor`` does real
array math that we want to verify.
"""

import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Import-time stubs (installed once, at collection time, before app import)
# ---------------------------------------------------------------------------

def _make_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


class _FakeSignal:
    """Stand-in for PySide6 Signal usable as a class attribute."""

    def __init__(self, *args, **kwargs):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args, **kwargs):
        for slot in list(self._slots):
            slot(*args, **kwargs)


class _StubBase:
    """Generic permissive base class for the stubbed Qt widgets/objects."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        # Any attribute access returns a no-op callable so method calls made
        # inside app code (setMinimumSize, update, ...) never explode.
        return lambda *a, **k: None


class _AutoAttr:
    """Namespace whose attribute access always succeeds (e.g. Qt.AlignCenter)."""

    def __getattr__(self, _name):
        return 0


def _install_stubs() -> None:
    # --- environment / api clients -----------------------------------------
    dotenv = _make_module("dotenv")
    dotenv.load_dotenv = lambda *a, **k: False

    groq = _make_module("groq")
    groq.Groq = _StubBase

    # --- audio / speech / tts ----------------------------------------------
    whisper = _make_module("whisper")
    whisper.load_model = lambda *a, **k: _StubBase()

    sd = _make_module("sounddevice")
    sd.rec = lambda *a, **k: None
    sd.wait = lambda *a, **k: None
    sd.stop = lambda *a, **k: None
    sd.InputStream = _StubBase

    _make_module("soundfile")

    edge_tts = _make_module("edge_tts")
    edge_tts.Communicate = _StubBase

    pygame = _make_module("pygame")
    mixer = types.SimpleNamespace(
        init=lambda *a, **k: None,
        music=types.SimpleNamespace(
            load=lambda *a, **k: None,
            play=lambda *a, **k: None,
            stop=lambda *a, **k: None,
            unload=lambda *a, **k: None,
            get_busy=lambda *a, **k: False,
        ),
    )
    pygame.mixer = mixer

    # --- PySide6 (only used as base classes / class-body constants) --------
    pyside6 = _make_module("PySide6")

    qtcore = _make_module("PySide6.QtCore")
    qtcore.Qt = _AutoAttr()
    qtcore.QTimer = _StubBase
    qtcore.QThread = _StubBase
    qtcore.Signal = lambda *a, **k: _FakeSignal()
    qtcore.Slot = lambda *a, **k: (lambda fn: fn)
    qtcore.QObject = _StubBase
    qtcore.QRectF = _StubBase
    qtcore.QPointF = _StubBase

    qtgui = _make_module("PySide6.QtGui")
    for _name in (
        "QPainter", "QColor", "QPen", "QBrush", "QFont",
        "QRadialGradient", "QLinearGradient", "QPainterPath", "QPolygonF",
    ):
        setattr(qtgui, _name, _StubBase)

    qtwidgets = _make_module("PySide6.QtWidgets")
    for _name in (
        "QApplication", "QMainWindow", "QWidget", "QLabel", "QVBoxLayout",
        "QHBoxLayout", "QFrame", "QSizePolicy", "QTextEdit",
    ):
        setattr(qtwidgets, _name, _StubBase)

    pyside6.QtCore = qtcore
    pyside6.QtGui = qtgui
    pyside6.QtWidgets = qtwidgets


_install_stubs()

# Import the application module *after* stubs are in place so its top-level
# ``import`` statements resolve to the stubs above.
import star_sara_v3 as app  # noqa: E402


@pytest.fixture
def app_module():
    """The imported star_sara_v3 module."""
    return app


@pytest.fixture
def data_paths(tmp_path, monkeypatch):
    """Redirect the module's JSON file paths into a throwaway temp directory.

    ``AIEngine.__init__`` reads (and may write) the on-disk JSON stores via
    these module-level globals, so pointing them at ``tmp_path`` keeps every
    test isolated and prevents clobbering the repo's real data files.
    """
    paths = {
        "USER_FILE": tmp_path / "user_data.json",
        "MEMORY_FILE": tmp_path / "memory.json",
        "TASKS_FILE": tmp_path / "tasks.json",
        "NOTES_FILE": tmp_path / "notes.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(app, name, str(path))
    return paths


@pytest.fixture
def ai(data_paths):
    """A fresh AIEngine backed by isolated temp JSON files."""
    return app.AIEngine()
