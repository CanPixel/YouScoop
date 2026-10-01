// On-device PCM capture with utterance endpoint detection. No browser speech service.
export class VoiceCapture {
  private stream: MediaStream;
  private context: AudioContext;
  private node: AudioWorkletNode;
  private frames: Float32Array[] = [];
  private voiced = false;
  private silence = 0;
  private duration = 0;
  private lead: Float32Array[] = [];
  private closed = false;
  private constructor(stream: MediaStream, context: AudioContext, node: AudioWorkletNode, private canListen: () => boolean, private onUtterance: (pcm: Float32Array) => void, private onLevel: (level: number) => void) {
    this.stream = stream; this.context = context; this.node = node;
    node.port.onmessage = (event: MessageEvent<Float32Array>) => this.chunk(event.data);
  }
  static async start(canListen: () => boolean, onUtterance: (pcm: Float32Array) => void, onLevel: (level: number) => void) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }, video: false });
    const context = new AudioContext();
    try {
      await context.audioWorklet.addModule(new URL("./voice-worklet.js", import.meta.url).href);
      const node = new AudioWorkletNode(context, "sacri-capture");
      context.createMediaStreamSource(stream).connect(node);
      const mute = context.createGain(); mute.gain.value = 0; node.connect(mute).connect(context.destination);
      await context.resume();
      return new VoiceCapture(stream, context, node, canListen, onUtterance, onLevel);
    } catch (error) { stream.getTracks().forEach((t) => t.stop()); await context.close(); throw error; }
  }
  private chunk(frame: Float32Array) {
    if (this.closed) return;
    const seconds = frame.length / this.context.sampleRate;
    const rms = Math.sqrt(frame.reduce((sum, n) => sum + n * n, 0) / frame.length);
    this.onLevel(Math.min(1, rms * 10));
    if (!this.canListen()) { this.reset(); return; }
    if (!this.voiced) {
      this.lead.push(frame); if (this.lead.length * seconds > .3) this.lead.shift();
      if (rms < .018) return;
      this.voiced = true; this.frames = [...this.lead]; this.lead = [];
    } else { this.frames.push(frame); }
    this.duration += seconds; this.silence = rms < .012 ? this.silence + seconds : 0;
    if (this.duration >= 29 || this.silence >= .8 && this.duration >= .5) {
      const combined = new Float32Array(this.frames.reduce((n, f) => n + f.length, 0));
      let position = 0; for (const f of this.frames) { combined.set(f, position); position += f.length; }
      const size = Math.min(480000, Math.floor(combined.length * 16000 / this.context.sampleRate));
      const pcm = new Float32Array(size);
      // Average the input window to reduce aliasing while downsampling to Whisper's 16 kHz.
      for (let i = 0; i < size; i++) {
        const a = Math.floor(i * this.context.sampleRate / 16000), b = Math.max(a + 1, Math.floor((i + 1) * this.context.sampleRate / 16000));
        let sum = 0; for (let j = a; j < b && j < combined.length; j++) sum += combined[j]; pcm[i] = sum / (b - a);
      }
      this.reset(); if (pcm.length >= 8000) this.onUtterance(pcm);
    }
  }
  private reset() { this.frames = []; this.lead = []; this.voiced = false; this.duration = 0; this.silence = 0; }
  async close() { this.closed = true; this.node.port.onmessage = null; this.node.disconnect(); this.stream.getTracks().forEach((track) => track.stop()); await this.context.close(); this.reset(); }
}
