import { expect, test } from "bun:test";
import { SpeechQueue, spokenText } from "../src/speech-queue";

// A fake native voice: each utterance resolves when finish() is called or stop() interrupts it.
function voice() {
  const spoken: string[] = []; const states: boolean[] = []; let finish: (() => void) | null = null;
  const queue = new SpeechQueue(
    (text) => new Promise<void>((resolve) => { spoken.push(text); finish = resolve; }),
    async () => { finish?.(); finish = null; },
    (speaking) => states.push(speaking),
    (error) => { throw error; },
  );
  const tick = () => new Promise((r) => setTimeout(r, 0));
  return { queue, spoken, states, tick, end: async () => { const f = finish; finish = null; f?.(); await tick(); } };
}

test("citations are not read aloud", () => {
  expect(spokenText("It was Baal [E1] [E2]. Unknown (unavailable evidence reference).")).toBe("It was Baal. Unknown.");
  expect(spokenText("This is reinforced in [E7], where **Baal** is named \"twice\" (not verbatim in the cited passages).")).toBe("This is reinforced, where Baal is named \"twice\".");
});

test("sentences queued during generation are spoken in order, one at a time", async () => {
  const v = voice();
  v.queue.push("First [E1]."); v.queue.push("Second."); await v.tick();
  expect(v.spoken).toEqual(["First."]);
  await v.end(); expect(v.spoken).toEqual(["First.", "Second."]);
  await v.end(); expect(v.queue.speaking).toBe(false);
  expect(v.states).toEqual([true, false]);
  v.queue.push("Third."); await v.tick();
  expect(v.spoken.at(-1)).toBe("Third."); expect(v.states).toEqual([true, false, true]);
});

test("a new answer after cancellation is not stranded behind the interrupted loop", async () => {
  const v = voice();
  v.queue.push("Old one."); v.queue.push("Old two."); await v.tick();
  await v.queue.cancel();
  v.queue.push("New one."); await v.tick(); await v.tick();
  expect(v.spoken).toEqual(["Old one.", "New one."]);
  expect(v.queue.speaking).toBe(true);
  await v.end(); expect(v.queue.speaking).toBe(false);
  expect(v.states).toEqual([true, false, true, false]);
});
