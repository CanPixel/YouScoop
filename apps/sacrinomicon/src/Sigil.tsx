import { useEffect, useRef } from "react";

export type SigilState = "idle" | "listening" | "thinking" | "speaking";

// The seven classical planets, set around the outer ring.
const PLANETS = ["☉", "☽", "☿", "♀", "♂", "♃", "♄"];
const ticks = Array.from({ length: 72 }, (_, i) => i);
const point = (r: number, angle: number) => [100 + r * Math.cos(angle - Math.PI / 2), 100 + r * Math.sin(angle - Math.PI / 2)] as const;
const polygon = (r: number, sides: number, turn = 0) => Array.from({ length: sides }, (_, i) => point(r, turn + i * Math.PI * 2 / sides).join(",")).join(" ");
// A heptagram {7/3}: every third vertex of a heptagon.
const heptagram = Array.from({ length: 7 }, (_, i) => point(58, (i * 3 % 7) * Math.PI * 2 / 7).join(",")).join(" ");

/**
 * An astrolabe-like seal. `level` is read every frame (0 to 1) so the seal
 * can breathe with a voice without re-rendering React.
 */
export default function Sigil({ state = "idle", level, size = 180, label }: { state?: SigilState; level?: () => number; size?: number; label?: string }) {
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!level) return;
    let frame = 0, smooth = 0;
    const tick = () => {
      smooth += (level() - smooth) * 0.25;
      root.current?.style.setProperty("--level", smooth.toFixed(3));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [level]);
  return <div ref={root} className={`sigil sigil-${state}`} style={{ width: size, height: size }} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
    <svg viewBox="0 0 200 200">
      <defs>
        <radialGradient id="sigil-core" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="var(--sigil-core)" stopOpacity=".55" />
          <stop offset="60%" stopColor="var(--sigil-core)" stopOpacity=".08" />
          <stop offset="100%" stopColor="var(--sigil-core)" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle className="sigil-aura" cx="100" cy="100" r="92" fill="url(#sigil-core)" />
      <g className="sigil-outer">
        <circle cx="100" cy="100" r="94" />
        <circle cx="100" cy="100" r="86" className="faint" />
        {ticks.map((i) => { const [x1, y1] = point(i % 6 ? 90 : 87, i * Math.PI / 36); const [x2, y2] = point(94, i * Math.PI / 36); return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} className={i % 6 ? "faint" : ""} />; })}
      </g>
      <g className="sigil-planets">
        {PLANETS.map((glyph, i) => { const [x, y] = point(76, i * Math.PI * 2 / 7); return <text key={glyph} x={x} y={y} textAnchor="middle" dominantBaseline="central">{glyph}</text>; })}
      </g>
      <g className="sigil-star">
        <circle cx="100" cy="100" r="66" className="faint" />
        <polygon points={heptagram} />
        <polygon points={polygon(58, 7)} className="faint" />
      </g>
      <g className="sigil-inner">
        <polygon points={polygon(30, 3)} />
        <polygon points={polygon(30, 3, Math.PI)} />
        <circle cx="100" cy="100" r="30" className="faint" />
      </g>
      <g className="sigil-eye">
        <path d="M78 100 Q100 84 122 100 Q100 116 78 100 Z" />
        <circle cx="100" cy="100" r="6.5" className="pupil" />
      </g>
    </svg>
  </div>;
}

/** A small static seal for the brand and message marks. */
export function SigilMark({ size = 22 }: { size?: number }) {
  return <svg className="sigil-mark" width={size} height={size} viewBox="0 0 200 200" aria-hidden="true">
    <circle cx="100" cy="100" r="90" />
    <polygon points={heptagram.split(" ").map((p) => p.split(",").map((n) => 100 + (Number(n) - 100) * 1.35).join(",")).join(" ")} />
    <path d="M70 100 Q100 78 130 100 Q100 122 70 100 Z" />
    <circle cx="100" cy="100" r="9" className="pupil" />
  </svg>;
}
