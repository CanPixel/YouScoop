import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import logo from "./assets/logo.png";
import { api, events } from "./api";
import SetupGate from "./components/SetupGate";
import Sidebar from "./components/Sidebar";
import VideoPane, { type VideoPaneHandle } from "./components/VideoPane";
import ChatPane from "./components/ChatPane";
import type {
  ChatMessage,
  ChunkPublic,
  IngestProgress,
  ModelProgress,
  SetupStatus,
  VideoMeta,
} from "./types";

const REQUIRED_MODELS = ["vlm", "mmproj", "embed"];
const CHATS_KEY = "smoltalk-chats-v1";

function loadChats(): Record<string, ChatMessage[]> {
  try {
    return JSON.parse(localStorage.getItem(CHATS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

let nextId = 0;
const uid = () => `${Date.now()}-${nextId++}`;

export default function App() {
  const [setup, setSetup] = useState<SetupStatus | null>(null);
  const [modelProgress, setModelProgress] = useState<Record<string, ModelProgress>>({});
  const [engineError, setEngineError] = useState<string | null>(null);

  const [videos, setVideos] = useState<VideoMeta[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [chunks, setChunks] = useState<Record<string, ChunkPublic[]>>({});
  const [ingest, setIngest] = useState<Record<string, IngestProgress>>({});

  const [chats, setChats] = useState<Record<string, ChatMessage[]>>(loadChats);
  const [generatingFor, setGeneratingFor] = useState<string | null>(null);

  const [addingUrl, setAddingUrl] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  const videoPaneRef = useRef<VideoPaneHandle>(null);
  const engineLoadRequested = useRef(false);
  const selectedIdRef = useRef<string | null>(null);
  selectedIdRef.current = selectedId;

  const refreshSetup = useCallback(async () => {
    try {
      setSetup(await api.setupStatus());
    } catch (e) {
      console.error("setup_status failed", e);
    }
  }, []);

  const refreshLibrary = useCallback(async () => {
    try {
      setVideos(await api.library());
    } catch (e) {
      console.error("library failed", e);
    }
  }, []);

  const fetchDetail = useCallback(async (id: string) => {
    try {
      const detail = await api.videoDetail(id);
      setChunks((prev) => ({ ...prev, [id]: detail.chunks }));
      setVideos((prev) => prev.map((v) => (v.id === id ? detail.meta : v)));
    } catch (e) {
      console.error("video_detail failed", e);
    }
  }, []);

  // ---- boot ----
  useEffect(() => {
    void refreshSetup();
    void refreshLibrary();

    const subs = [
      events.onModelProgress((p) => {
        setModelProgress((prev) => ({ ...prev, [p.id]: p }));
        if (p.done) void refreshSetup();
      }),
      events.onEngineStatus((s) => {
        setEngineError(s.error ?? null);
        void refreshSetup();
      }),
      events.onIngestProgress((p) => {
        if (!p.videoId) {
          // Global (pre-id) events: metadata fetch or an early failure.
          if (p.stage === "error") {
            setAddError(p.detail);
            setAddingUrl(false);
          }
          return;
        }
        setIngest((prev) => ({ ...prev, [p.videoId]: p }));
        if (p.meta) {
          setAddingUrl(false);
          setAddError(null);
          const meta = p.meta;
          setVideos((prev) => {
            const exists = prev.some((v) => v.id === meta.id);
            return exists ? prev.map((v) => (v.id === meta.id ? meta : v)) : [meta, ...prev];
          });
          if (p.stage === "metadata" && !selectedIdRef.current) {
            setSelectedId(meta.id);
          }
        }
        if (p.stage === "ready") {
          void fetchDetail(p.videoId);
        }
      }),
      events.onChatToken((t) => {
        setChats((prev) => {
          const msgs = prev[t.videoId];
          if (!msgs || msgs.length === 0) return prev;
          const last = msgs[msgs.length - 1];
          if (last.role !== "assistant" || !last.streaming) return prev;
          const updated = [...msgs.slice(0, -1), { ...last, text: last.text + t.token }];
          return { ...prev, [t.videoId]: updated };
        });
      }),
    ];
    return () => {
      for (const s of subs) void s.then((un) => un());
    };
  }, [refreshSetup, refreshLibrary, fetchDetail]);

  // ---- auto-load engine once required models exist ----
  useEffect(() => {
    if (!setup || setup.engineLoaded || setup.engineLoading || engineLoadRequested.current) return;
    const ready = setup.models
      .filter((m) => REQUIRED_MODELS.includes(m.id))
      .every((m) => m.downloaded);
    if (ready) {
      engineLoadRequested.current = true;
      api.loadEngine().catch((e) => {
        engineLoadRequested.current = false;
        setEngineError(String(e));
      });
    }
  }, [setup]);

  // ---- persist chats ----
  useEffect(() => {
    try {
      localStorage.setItem(CHATS_KEY, JSON.stringify(chats));
    } catch {
      // best effort
    }
  }, [chats]);

  // ---- fetch chunks when selecting a ready video ----
  useEffect(() => {
    if (!selectedId) return;
    const v = videos.find((x) => x.id === selectedId);
    if (v?.status === "ready" && !chunks[selectedId]) {
      void fetchDetail(selectedId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, videos]);

  const addVideo = useCallback(
    (url: string) => {
      setAddingUrl(true);
      setAddError(null);
      api.addVideo(url).catch((e) => {
        setAddingUrl(false);
        setAddError(String(e));
      });
    },
    [],
  );

  const retryVideo = useCallback(
    (id: string) => {
      // Optimistic UI: drop the stale ingest record and flip the entry back to
      // processing; the pipeline's events take over from here.
      setIngest((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setVideos((prev) =>
        prev.map((v) => (v.id === id ? { ...v, status: "processing", error: null } : v)),
      );
      api.retryVideo(id).catch((e) => {
        setAddError(String(e));
        void refreshLibrary();
      });
    },
    [refreshLibrary],
  );

  const removeVideo = useCallback(
    async (id: string) => {
      // Confirmation is the two-step button in the sidebar: window.confirm()
      // is a no-op in Tauri's webview (always returns falsy).
      try {
        await api.removeVideo(id);
      } catch (e) {
        console.error(e);
      }
      setVideos((prev) => prev.filter((v) => v.id !== id));
      setChats((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      if (selectedIdRef.current === id) setSelectedId(null);
    },
    [],
  );

  const sendQuestion = useCallback(
    async (videoId: string, question: string) => {
      const history: { question: string; answer: string }[] = [];
      const msgs = chats[videoId] ?? [];
      for (let i = 0; i < msgs.length - 1; i++) {
        if (msgs[i].role === "user" && msgs[i + 1].role === "assistant" && !msgs[i + 1].error) {
          history.push({ question: msgs[i].text, answer: msgs[i + 1].text });
        }
      }

      const userMsg: ChatMessage = { id: uid(), role: "user", text: question };
      const draft: ChatMessage = { id: uid(), role: "assistant", text: "", streaming: true };
      setChats((prev) => ({ ...prev, [videoId]: [...(prev[videoId] ?? []), userMsg, draft] }));
      setGeneratingFor(videoId);

      try {
        const answer = await api.askVideo(videoId, question, history.slice(-2));
        setChats((prev) => {
          const list = prev[videoId] ?? [];
          const updated = list.map((m) =>
            m.id === draft.id ? { ...m, text: answer, streaming: false } : m,
          );
          return { ...prev, [videoId]: updated };
        });
      } catch (e) {
        setChats((prev) => {
          const list = prev[videoId] ?? [];
          const updated = list.map((m) =>
            m.id === draft.id
              ? { ...m, text: String(e), streaming: false, error: true }
              : m,
          );
          return { ...prev, [videoId]: updated };
        });
      } finally {
        setGeneratingFor(null);
      }
    },
    [chats],
  );

  const selected = useMemo(
    () => videos.find((v) => v.id === selectedId) ?? null,
    [videos, selectedId],
  );

  const needsSetup =
    !setup ||
    !setup.engineLoaded ||
    !setup.models.filter((m) => REQUIRED_MODELS.includes(m.id)).every((m) => m.downloaded);

  if (!setup) {
    return (
      <div className="boot-screen">
        <span className="spinner" />
      </div>
    );
  }

  if (needsSetup) {
    return (
      <SetupGate
        status={setup}
        progress={modelProgress}
        engineError={engineError}
        onRefresh={refreshSetup}
      />
    );
  }

  return (
    <div className="app-shell">
      <Sidebar
        videos={videos}
        selectedId={selectedId}
        ingest={ingest}
        addingUrl={addingUrl}
        addError={addError}
        onSelect={setSelectedId}
        onAdd={addVideo}
        onRetry={retryVideo}
        onRemove={(id) => void removeVideo(id)}
      />

      {selected ? (
        <main className="workspace">
          <VideoPane
            ref={videoPaneRef}
            meta={selected}
            chunks={chunks[selected.id] ?? []}
            ingest={ingest[selected.id]}
            onCancelIngest={() => void api.cancelIngest(selected.id)}
            onRetry={() => retryVideo(selected.id)}
          />
          <ChatPane
            meta={selected}
            messages={chats[selected.id] ?? []}
            generating={generatingFor === selected.id}
            onSend={(q) => void sendQuestion(selected.id, q)}
            onStop={() => void api.stopChat()}
            onSeek={(secs) => videoPaneRef.current?.seekTo(secs)}
          />
        </main>
      ) : (
        <main className="workspace welcome">
          <div className="welcome-card">
            <img src={logo} alt="" className="welcome-logo" />
            <h1>SmolTalk</h1>
            <p>
              Paste a YouTube link in the sidebar. SmolTalk downloads the video, reads the
              transcript, <em>watches</em> the frames with SmolVLM2, and builds a local semantic
              index. Then you can talk to the video!
            </p>
            <div className="welcome-steps">
              <div><span>1</span> Add a YouTube URL</div>
              <div><span>2</span> Wait for the local indexing pass</div>
              <div><span>3</span> Ask anything, click timestamps to jump</div>
            </div>
            <p className="welcome-privacy">100% on-device · no accounts · no API keys</p>
          </div>
        </main>
      )}
    </div>
  );
}
