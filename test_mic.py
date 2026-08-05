import sounddevice as sd
import soundfile as sf

print("Recording for 5 seconds...")

audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1
)

sd.wait()

sf.write(
    "test.wav",
    audio,
    16000
)

print("Saved test.wav")