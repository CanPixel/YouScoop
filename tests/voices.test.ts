import { expect, test } from "bun:test";
import { visibleVoices } from "../src/lib/voices";

test("a brief guest remains labeled alongside the main presenter", () => {
  expect([...visibleVoices([
    { voice: "Voice A", seconds: 1859 },
    { voice: "Voice B", seconds: 67 },
  ])]).toEqual(["Voice A", "Voice B"]);
});

test("small noisy clusters do not turn a single narrator into multiple speakers", () => {
  expect(visibleVoices([
    { voice: "Voice A", seconds: 1859 },
    { voice: "Voice B", seconds: 4.9 },
  ]).size).toBe(0);
});

test("short videos retain speakers at the five-second threshold", () => {
  expect([...visibleVoices([
    { voice: "Voice A", seconds: 12 },
    { voice: "Voice B", seconds: 5 },
    { voice: "Voice C", seconds: 0.7 },
  ])]).toEqual(["Voice A", "Voice B"]);
  expect(visibleVoices([]).size).toBe(0);
});
