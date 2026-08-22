import React from "react";
import { parseTs } from "../lib/format";

interface Props {
  text: string;
  onSeek: (secs: number) => void;
}

/**
 * Minimal markdown-ish renderer: paragraphs, bullet lists, **bold**, `code`,
 * and clickable [m:ss] / [h:mm:ss] timestamp citations.
 */
export default function MessageBody({ text, onSeek }: Props) {
  const blocks = text.split(/\n{2,}/);
  return (
    <div className="msg-body">
      {blocks.map((block, i) => {
        const lines = block.split("\n").filter((l) => l.trim().length > 0);
        const isList = lines.length > 0 && lines.every((l) => /^\s*[-*•]\s+/.test(l));
        if (isList) {
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>{renderInline(l.replace(/^\s*[-*•]\s+/, ""), onSeek)}</li>
              ))}
            </ul>
          );
        }
        return <p key={i}>{lines.map((l, j) => (
          <React.Fragment key={j}>
            {j > 0 && <br />}
            {renderInline(l, onSeek)}
          </React.Fragment>
        ))}</p>;
      })}
    </div>
  );
}

const TS = String.raw`\d{1,2}:\d{2}(?::\d{2})?`;
// The model cites either one timestamp or a comma-separated run of them
// ("[4:52, 5:18, 9:00]"), and both should come out as chips. Every alternative
// stays non-capturing: String.split() emits each capture group it sees, so an
// extra group here would leak fragments into the output.
const INLINE_RE = new RegExp(
  `(\\[\\s*${TS}(?:\\s*,\\s*${TS})*\\s*\\]|\\*\\*[^*]+\\*\\*|\`[^\`]+\`)`,
  "g",
);
const TS_GROUP_RE = new RegExp(`^\\[\\s*(${TS}(?:\\s*,\\s*${TS})*)\\s*\\]$`);

function renderInline(text: string, onSeek: (secs: number) => void): React.ReactNode[] {
  const parts = text.split(INLINE_RE);
  return parts.map((part, i) => {
    if (!part) return null;
    const tsMatch = part.match(TS_GROUP_RE);
    if (tsMatch) {
      const stamps = tsMatch[1].split(",").map((t) => t.trim());
      const chips = stamps
        .map((label) => ({ label, secs: parseTs(label) }))
        .filter((c): c is { label: string; secs: number } => c.secs !== null);
      // Fall back to the raw text rather than dropping a citation we can't
      // fully parse — a half-rendered list would be worse than plain text.
      if (chips.length === stamps.length && chips.length > 0) {
        return (
          <React.Fragment key={i}>
            {chips.map((c) => (
              <button
                key={c.label}
                className="ts-chip"
                onClick={() => onSeek(c.secs)}
                title={`Jump to ${c.label}`}
              >
                {c.label}
              </button>
            ))}
          </React.Fragment>
        );
      }
      return part;
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}
