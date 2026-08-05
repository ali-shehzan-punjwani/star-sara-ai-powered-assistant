// Captures mono float32 audio from the mic graph, converts to PCM16 and posts
// it to the main thread in ~30 ms frames (the frame size the backend VAD wants).
class PcmRecorder extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const targetRate = options?.processorOptions?.targetSampleRate ?? 16000;
    this.ratio = sampleRate / targetRate;
    this.frameSize = Math.round(targetRate * 0.03);
    this.buffer = new Float32Array(0);
    this.position = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;

    // Linear resample to the target rate.
    const out = [];
    while (this.position < input.length) {
      out.push(input[Math.floor(this.position)]);
      this.position += this.ratio;
    }
    this.position -= input.length;

    const merged = new Float32Array(this.buffer.length + out.length);
    merged.set(this.buffer);
    merged.set(out, this.buffer.length);
    this.buffer = merged;

    while (this.buffer.length >= this.frameSize) {
      const frame = this.buffer.subarray(0, this.frameSize);
      this.buffer = this.buffer.slice(this.frameSize);

      const pcm = new Int16Array(frame.length);
      let peak = 0;
      for (let i = 0; i < frame.length; i += 1) {
        const sample = Math.max(-1, Math.min(1, frame[i]));
        pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        peak = Math.max(peak, Math.abs(sample));
      }
      this.port.postMessage({ pcm: pcm.buffer, peak }, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-recorder", PcmRecorder);
