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

const INLINE_RE = /(\[\d{1,2}:\d{2}(?::\d{2})?\]|\*\*[^*]+\*\*|`[^`]+`)/g;

function renderInline(text: string, onSeek: (secs: number) => void): React.ReactNode[] {
  const parts = text.split(INLINE_RE);
  return parts.map((part, i) => {
    if (!part) return null;
    const tsMatch = part.match(/^\[(\d{1,2}:\d{2}(?::\d{2})?)\]$/);
    if (tsMatch) {
      const secs = parseTs(tsMatch[1]);
      if (secs !== null) {
        return (
          <button key={i} className="ts-chip" onClick={() => onSeek(secs)} title={`Jump to ${tsMatch[1]}`}>
            {tsMatch[1]}
          </button>
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
