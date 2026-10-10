import type { VoiceSummary } from "../types";

/** Keep short guest appearances; ignore clusters with less than five seconds. */
export function visibleVoices(voices: VoiceSummary[]): Set<string> {
  const distinct = voices.filter((v) => v.seconds >= 5);
  return new Set(distinct.length >= 2 ? distinct.map((v) => v.voice) : []);
}
