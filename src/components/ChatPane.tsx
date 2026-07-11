import { useCallback, useEffect, useRef, useState } from "react";
import MessageBody from "./MessageBody";
import type { ChatMessage, VideoMeta } from "../types";

interface Props {
  meta: VideoMeta;
  messages: ChatMessage[];
  generating: boolean;
  onSend: (question: string) => void;
  onStop: () => void;
  onSeek: (secs: number) => void;
}

const SUGGESTIONS = [
  "Summarize this video in a few sentences",
  "What are the key moments?",
  "What is shown on screen at the start?",
  "What conclusions does it reach?",
];

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // WKWebView fallback
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  }
}

export default function ChatPane({ meta, messages, generating, onSend, onStop, onSeek }: Props) {
  const [input, setInput] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const ready = meta.status === "ready";

  // Focus the input as soon as the video becomes chattable / is selected.
  useEffect(() => {
    if (ready && !generating) inputRef.current?.focus();
  }, [ready, generating, meta.id]);

  // Grow the textarea with its content (up to the CSS max-height).
  const autosize = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, []);

  useEffect(autosize, [input, autosize]);

  // Esc anywhere stops the in-flight generation.
  useEffect(() => {
    if (!generating) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onStop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [generating, onStop]);

  const send = (text: string) => {
    const q = text.trim();
    if (!q || generating || !ready) return;
    onSend(q);
    setInput("");
    inputRef.current?.focus();
  };

  const copyMessage = async (m: ChatMessage) => {
    if (await copyText(m.text)) {
      setCopiedId(m.id);
      setTimeout(() => setCopiedId((prev) => (prev === m.id ? null : prev)), 1500);
    }
  };

  return (
    <section className="chat-pane">
      <header className="chat-head">
        <div className="chat-head-title">Chat</div>
        <div className="chat-head-sub">
          {ready
            ? `${meta.chunkCount} indexed moments · Qwen3 4B`
            : "Available once processing finishes"}
        </div>
      </header>

      <div className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="chat-empty-glyph">💬</div>
            <p className="chat-empty-text">
              {ready
                ? "Ask anything about this video. Answers cite clickable timestamps."
                : "Once the video is indexed you can ask questions about anything said or shown."}
            </p>
            {ready && (
              <div className="chat-suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="suggestion" onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map((m) => (
            <div key={m.id} className={`msg-row ${m.role} ${m.error ? "error" : ""}`}>
              <div className="msg-bubble">
                {m.text ? (
                  <MessageBody text={m.text} onSeek={onSeek} />
                ) : m.streaming ? (
                  <span className="typing">
                    <i /><i /><i />
                  </span>
                ) : null}
                {m.streaming && m.text && <span className="caret" />}
                {m.role === "assistant" && !m.streaming && m.text && (
                  <button
                    className={`msg-copy ${copiedId === m.id ? "copied" : ""}`}
                    title="Copy response"
                    onClick={() => void copyMessage(m)}
                  >
                    {copiedId === m.id ? "✓ Copied" : "⧉ Copy"}
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      <div className="chat-input-area">
        {generating && (
          <button className="btn btn-ghost btn-sm chat-stop" onClick={onStop}>
            ■ Stop generating
          </button>
        )}
        <form
          className="chat-form"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder={ready ? "Ask about the video…" : "Waiting for processing…"}
            value={input}
            rows={1}
            disabled={!ready || generating}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
          />
          <button
            type="submit"
            className="btn btn-primary chat-send"
            disabled={!ready || generating || !input.trim()}
            title="Send"
          >
            ↑
          </button>
        </form>
      </div>
    </section>
  );
}
