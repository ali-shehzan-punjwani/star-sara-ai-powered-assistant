"""Pre-download the openWakeWord ONNX models so the first run is not delayed."""

from __future__ import annotations


def main() -> None:
    try:
        from openwakeword.utils import download_models
    except ImportError:
        print("openwakeword is not installed; skipping.")
        return
    download_models()
    print("openWakeWord models ready.")


if __name__ == "__main__":
    main()
