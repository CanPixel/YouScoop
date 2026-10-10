// The Oracle's voice in the room: rendered sentences pass through a slight
// pitch drop, a warm low shelf and softened highs, dry and close.
// An analyser exposes the voice's level so the sigil can breathe with it.

/** Playback rate below 1 lowers pitch; the renderer speaks faster to compensate. */
const DEPTH = 0.95;

export class VoiceStage {
  private context: AudioContext | null = null;
  private input: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private samples = new Float32Array(1024);
  private current: AudioBufferSourceNode | null = null;
  private finish: (() => void) | null = null;

  private ensure() {
    if (this.context) return this.context;
    const context = new AudioContext();
    const input = context.createGain();
    const warmth = context.createBiquadFilter(); warmth.type = "lowshelf"; warmth.frequency.value = 190; warmth.gain.value = 4.5;
    const body = context.createBiquadFilter(); body.type = "peaking"; body.frequency.value = 420; body.Q.value = 0.9; body.gain.value = 1.5;
    const soften = context.createBiquadFilter(); soften.type = "highshelf"; soften.frequency.value = 6500; soften.gain.value = -5;
    const presence = context.createBiquadFilter(); presence.type = "peaking"; presence.frequency.value = 2600; presence.Q.value = 1.2; presence.gain.value = -1.5;
    const compressor = context.createDynamicsCompressor();
    compressor.threshold.value = -20; compressor.ratio.value = 3; compressor.attack.value = 0.008; compressor.release.value = 0.25;
    const master = context.createGain(); master.gain.value = 0.95;
    const analyser = context.createAnalyser(); analyser.fftSize = 1024; analyser.smoothingTimeConstant = 0.6;
    input.connect(warmth).connect(body).connect(presence).connect(soften).connect(compressor);
    compressor.connect(master);
    master.connect(analyser).connect(context.destination);
    this.context = context; this.input = input; this.analyser = analyser;
    return context;
  }

  /** Decode a rendered sentence ahead of playback. */
  async decode(bytes: ArrayBuffer) {
    return this.ensure().decodeAudioData(bytes);
  }

  /** Play one sentence; resolves when it has been heard or is stopped. */
  async play(buffer: AudioBuffer) {
    const context = this.ensure();
    if (context.state === "suspended") await context.resume();
    this.halt();
    const source = context.createBufferSource();
    source.buffer = buffer; source.playbackRate.value = DEPTH;
    source.connect(this.input!);
    this.current = source;
    await new Promise<void>((resolve) => {
      this.finish = resolve;
      source.onended = () => { if (this.current === source) this.current = null; this.finish = null; resolve(); };
      source.start();
    });
  }

  /** A soft struck-bowl tone that marks the Oracle beginning to listen. */
  chime() {
    const context = this.ensure();
    if (context.state === "suspended") void context.resume();
    const now = context.currentTime;
    for (const [frequency, gain, length] of [[196, 0.05, 2.4], [392.4, 0.025, 1.6], [588.9, 0.012, 1.1]] as const) {
      const tone = context.createOscillator(), level = context.createGain();
      tone.frequency.value = frequency; tone.type = "sine";
      level.gain.setValueAtTime(0.0001, now);
      level.gain.exponentialRampToValueAtTime(gain, now + 0.02);
      level.gain.exponentialRampToValueAtTime(0.0001, now + length);
      tone.connect(level).connect(this.input!);
      tone.start(now); tone.stop(now + length + 0.05);
    }
  }

  /** Current loudness of the voice, 0 to 1. */
  level() {
    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this.samples);
    let sum = 0; for (const s of this.samples) sum += s * s;
    return Math.min(1, Math.sqrt(sum / this.samples.length) * 6);
  }

  private halt() {
    const source = this.current; this.current = null;
    if (source) { source.onended = null; try { source.stop(); } catch { /* already ended */ } }
    const finish = this.finish; this.finish = null; finish?.();
  }

  stop() { this.halt(); }

  async close() { this.halt(); await this.context?.close(); this.context = null; this.input = null; this.analyser = null; }
}
