// Speaks Oracle sentences in order while the answer is still generating. Each
// sentence is prepared (rendered to audio) as soon as it arrives, so the next
// one is ready when the current one ends; playback stays strictly in order.
export const spokenText = (text: string) => text
  .replace(/\[E\d+\]|\((?:unavailable evidence reference|not verbatim in the cited passages|from an automatic frame description; wording may be invented)\)/g, "")
  .replace(/\*\*|__|[*_`#>]/g, "")
  // "as stated in [E2]," would otherwise be heard as "as stated in,".
  .replace(/\b(?:as (?:stated|shown|seen|noted) )?(?:in|by|from|per|see)\s+(?=[,.;:!?])/gi, "")
  .replace(/\s+—\s+|—/g, ", ")
  .replace(/\s+([.,;:!?])/g, "$1")
  .replace(/\s{2,}/g, " ")
  .trim();

export class SpeechQueue<T = string> {
  private queue: Promise<T>[] = [];
  private generation = 0;
  private active: number | null = null;
  constructor(
    private speak: (item: T) => Promise<unknown>,
    private stop: () => Promise<unknown>,
    private onSpeaking: (speaking: boolean) => void,
    private onError: (error: unknown) => void,
    private prepare: (text: string) => Promise<T> = async (text) => text as unknown as T,
  ) {}
  get speaking() { return this.active !== null; }
  push(text: string) {
    const spoken = spokenText(text);
    if (!spoken) return;
    const prepared = this.prepare(spoken);
    // A cancelled sentence may still fail; nobody is waiting for it.
    prepared.catch(() => {});
    this.queue.push(prepared);
    const generation = this.generation;
    if (this.active === generation) return;
    this.active = generation; this.onSpeaking(true);
    void this.drain(generation);
  }
  private async drain(generation: number) {
    try {
      while (this.queue.length && generation === this.generation) {
        const item = await this.queue.shift()!;
        if (generation !== this.generation) break;
        await this.speak(item);
      }
    } catch (error) { if (generation === this.generation) this.onError(error); }
    finally {
      // A cancelled loop leaves the state to the generation that replaced it.
      if (this.active === generation) { this.active = null; this.queue = []; this.onSpeaking(false); }
    }
  }
  async cancel() {
    this.generation++; this.queue = [];
    if (this.active !== null) { this.active = null; this.onSpeaking(false); }
    await this.stop();
  }
}
