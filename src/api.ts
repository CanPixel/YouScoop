import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  ChatDone,
  ChatToken,
  EngineStatus,
  IngestProgress,
  ModelProgress,
  SetupStatus,
  VideoDetail,
  VideoMeta,
} from "./types";

export const api = {
  setupStatus: () => invoke<SetupStatus>("setup_status"),
  downloadModel: (id: string) => invoke<void>("download_model", { id }),
  loadEngine: () => invoke<void>("load_engine"),
  library: () => invoke<VideoMeta[]>("library"),
  videoDetail: (id: string) => invoke<VideoDetail>("video_detail", { id }),
  addVideo: (url: string) => invoke<void>("add_video", { url }),
  retryVideo: (id: string) => invoke<void>("retry_video", { id }),
  cancelIngest: (id: string) => invoke<void>("cancel_ingest", { id }),
  removeVideo: (id: string) => invoke<void>("remove_video", { id }),
  askVideo: (videoId: string, question: string, history: { question: string; answer: string }[]) =>
    invoke<string>("ask_video", { videoId, question, history }),
  stopChat: () => invoke<void>("stop_chat"),
};

export const events = {
  onModelProgress: (cb: (p: ModelProgress) => void): Promise<UnlistenFn> =>
    listen<ModelProgress>("model-progress", (e) => cb(e.payload)),
  onEngineStatus: (cb: (p: EngineStatus) => void): Promise<UnlistenFn> =>
    listen<EngineStatus>("engine-status", (e) => cb(e.payload)),
  onIngestProgress: (cb: (p: IngestProgress) => void): Promise<UnlistenFn> =>
    listen<IngestProgress>("ingest-progress", (e) => cb(e.payload)),
  onChatToken: (cb: (p: ChatToken) => void): Promise<UnlistenFn> =>
    listen<ChatToken>("chat-token", (e) => cb(e.payload)),
  onChatDone: (cb: (p: ChatDone) => void): Promise<UnlistenFn> =>
    listen<ChatDone>("chat-done", (e) => cb(e.payload)),
};
