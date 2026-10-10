import type { ReactNode } from "react";
import { timestamp } from "./EvidenceDrawer";
import type { Evidence, SearchHit } from "./types";

const TAGS = /\((not verbatim in the cited passages|from an automatic frame description; wording may be invented|unavailable evidence reference)\)/;
const kindLabel = (kind: string) => kind === "document" ? "on-screen text" : kind;

/** A citation mark that previews its passage on hover or focus. */
function Citation({ n, hit, onOpen }: { n: number; hit: SearchHit; onOpen: (evidence: Evidence) => void }) {
  const text = hit.reading ?? hit.evidence.text;
  return <span className="citation-wrap">
    <button type="button" className="citation" aria-label={`Passage ${n}: ${hit.sourceTitle} at ${timestamp(hit.evidence.start)}`} onClick={() => onOpen(hit.evidence)}>{n}</button>
    <span className="citation-preview" role="tooltip">
      <span className="eyebrow">{timestamp(hit.evidence.start)} · {kindLabel(hit.evidence.kind)}</span>
      <strong>{hit.sourceTitle}</strong>
      <span>{text.length > 260 ? `${text.slice(0, 260).trim()}…` : text}</span>
    </span>
  </span>;
}

/** Bold, italics, citations and verification tags within one line. */
function inline(text: string, sources: SearchHit[] | undefined, onOpen: (evidence: Evidence) => void, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\[E\d+\]|\((?:not verbatim in the cited passages|from an automatic frame description; wording may be invented|unavailable evidence reference)\))/g;
  let last = 0, i = 0;
  for (const match of text.matchAll(pattern)) {
    const part = match[0], at = match.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    const k = `${key}-${i++}`;
    if (part.startsWith("**")) out.push(<strong key={k}>{part.slice(2, -2)}</strong>);
    else if (part.startsWith("*")) out.push(<em key={k}>{part.slice(1, -1)}</em>);
    else if (part.startsWith("[E")) {
      const n = Number(part.slice(2, -1)); const hit = sources?.[n - 1];
      out.push(hit ? <Citation key={k} n={n} hit={hit} onOpen={onOpen} /> : <span key={k} className="citation missing">{n}</span>);
    } else if (TAGS.test(part)) {
      const tag = part.slice(1, -1);
      out.push(<span key={k} className="verify-tag" title={tag}>{tag.startsWith("not verbatim") ? "not verbatim" : tag.startsWith("from an automatic") ? "from a frame description" : "missing passage"}</span>);
    }
    last = at + part.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** The Oracle's reply: paragraphs and lists, with live citations. */
export default function Answer({ text, sources, onOpen }: { text: string; sources?: SearchHit[]; onOpen: (evidence: Evidence) => void }) {
  const blocks = text.replace(/\r/g, "").split(/\n{2,}/).map((b) => b.trim()).filter(Boolean);
  return <div className="answer-body">{blocks.map((block, b) => {
    const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
    const bullet = /^(?:[-*•]|\d+[.)])\s+/;
    if (lines.length > 1 && lines.every((l) => bullet.test(l))) {
      const ordered = /^\d/.test(lines[0]);
      const items = lines.map((l, i) => <li key={i}>{inline(l.replace(bullet, ""), sources, onOpen, `${b}-${i}`)}</li>);
      return ordered ? <ol key={b}>{items}</ol> : <ul key={b}>{items}</ul>;
    }
    if (lines.length > 1 && lines.slice(1).every((l) => bullet.test(l))) {
      return <div key={b}><p>{inline(lines[0], sources, onOpen, `${b}-h`)}</p><ul>{lines.slice(1).map((l, i) => <li key={i}>{inline(l.replace(bullet, ""), sources, onOpen, `${b}-${i}`)}</li>)}</ul></div>;
    }
    return <p key={b}>{lines.flatMap((l, i) => i ? [<br key={`br${i}`} />, ...inline(l, sources, onOpen, `${b}-${i}`)] : inline(l, sources, onOpen, `${b}-${i}`))}</p>;
  })}</div>;
}
