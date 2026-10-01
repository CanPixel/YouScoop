import { useMemo, useState } from "react";
import type { Concept, Library } from "./types";

export default function ConceptMap({ library, selected, onSelect }: { library: Library; selected: string | null; onSelect: (concept: Concept) => void }) {
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const nodes = useMemo(() => {
    const chosen = library.concepts.find((c) => c.id === selected);
    const connected = new Set(library.relationships.filter((r) => r.fromId === selected || r.toId === selected).flatMap((r) => [r.fromId, r.toId]));
    const concepts = chosen ? [chosen, ...library.concepts.filter((c) => c.id !== selected && (connected.has(c.id) || c.evidenceIds.some((id) => chosen.evidenceIds.includes(id))))].slice(0, 24) : library.concepts.slice(0, 24);
    return concepts.map((c, i) => ({ ...c, x: i === 0 ? 400 : 400 + Math.cos((i - 1) * Math.PI * 2 / Math.max(1, concepts.length - 1)) * 250, y: i === 0 ? 260 : 260 + Math.sin((i - 1) * Math.PI * 2 / Math.max(1, concepts.length - 1)) * 190 }));
  }, [library, selected]);
  const edges = nodes.flatMap((a, i) => nodes.slice(i + 1).flatMap((b) => {
    const typed = library.relationships.find((r) => (r.fromId === a.id && r.toId === b.id) || (r.fromId === b.id && r.toId === a.id));
    const shared = a.evidenceIds.some((id) => b.evidenceIds.includes(id));
    return typed || shared ? [{ a, b, label: typed?.kind ?? "shared passage", typed: !!typed }] : [];
  }));
  return <div className="concept-map">
    <div className="map-controls"><button aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(.5, z - .15))}>−</button><button aria-label="Reset map" onClick={() => { setZoom(1); setOffset({ x: 0, y: 0 }); }}>Recenter</button><button aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(2, z + .15))}>+</button></div>
    <svg viewBox="0 0 800 520" aria-label="Concept relationships" onPointerDown={(e) => { if ((e.target as Element).tagName === "svg") { e.currentTarget.setPointerCapture(e.pointerId); setDrag({ x: e.clientX, y: e.clientY }); } }} onPointerMove={(e) => { if (drag) { setOffset((p) => ({ x: p.x + e.clientX - drag.x, y: p.y + e.clientY - drag.y })); setDrag({ x: e.clientX, y: e.clientY }); } }} onPointerUp={() => setDrag(null)}>
      <g transform={`translate(${offset.x} ${offset.y}) translate(400 260) scale(${zoom}) translate(-400 -260)`}>
        {edges.map(({ a, b, label, typed }) => <g key={`${a.id}-${b.id}`}><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={typed ? "typed-edge" : "shared-edge"} /><text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 8} className="edge-label">{label.replaceAll("_", " ")}</text></g>)}
        {nodes.map((node) => <g key={node.id} transform={`translate(${node.x} ${node.y})`} className={`concept-node ${node.id === selected ? "selected" : ""}`} role="button" tabIndex={0} aria-label={`Explore ${node.label}`} onClick={() => onSelect(node)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(node); } }}><circle r={node.id === selected ? 33 : 23} /><circle r={5} className="node-heart" /><text y={48} textAnchor="middle">{node.label}</text></g>)}
      </g>
    </svg>
    {!nodes.length && <div className="map-empty"><span>◇</span><p>Your constellation begins with a source.</p><small>Open a passage and name a concept. Shared passages and explicit relationships form the map.</small></div>}
    <p className="map-caption">Up to 24 nearby concepts · dashed links share a passage · solid links carry an explicit relationship</p>
  </div>;
}
