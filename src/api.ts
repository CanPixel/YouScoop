import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  ChatAnswer,
  ChatDone,
  ChatToken,
  EngineStatus,
  IngestProgress,
  ModelProgress,
  SetupStatus,
  VideoDetail,
  VideoMeta,
  SavedSource,
  WorkerStatus,
  WorkerConfig,
} from "./types";

export const api = {
  setupStatus: () => invoke<SetupStatus>("setup_status"),
  downloadModel: (id: string) => invoke<void>("download_model", { id }),
  updateYtdlp: () => invoke<string>("update_ytdlp"),
  loadEngine: () => invoke<void>("load_engine"),
  library: () => invoke<VideoMeta[]>("library"),
  videoDetail: (id: string) => invoke<VideoDetail>("video_detail", { id }),
  saveToArchive: (id: string) => invoke<SavedSource>("save_to_archive", { id }),
  addVideo: (url: string) => invoke<void>("add_video", { url }),
  retryVideo: (id: string) => invoke<void>("retry_video", { id }),
  updateAnalysis: (id: string) => invoke<void>("update_analysis", { id }),
  cancelIngest: (id: string) => invoke<void>("cancel_ingest", { id }),
  removeVideo: (id: string) => invoke<void>("remove_video", { id }),
  askVideo: (videoId: string, question: string, history: { question: string; answer: string }[]) =>
    invoke<ChatAnswer>("ask_video", { videoId, question, history }),
  stopChat: () => invoke<void>("stop_chat"),
  workerStatus: () => invoke<WorkerStatus>("worker_status"),
  configureWorker: (config: WorkerConfig) => invoke<void>("configure_worker", { config }),
  setupDesktopRuntime: () => invoke<WorkerStatus>("setup_desktop_runtime"),
  installDesktopModel: (model: "gemma", source: string | null) => invoke<WorkerStatus>("install_desktop_model", { model, source }),
  askGemma: (videoId: string, question: string, history: {question: string; answer: string}[]) => invoke<ChatAnswer>("ask_gemma", { videoId, question, history }),
  stopWorker: () => invoke<void>("stop_worker"),
};

export const events = {
  onRuntimeProgress: (cb: (p: { kind: string; detail?: string; received?: number; total?: number }) => void): Promise<UnlistenFn> =>
    listen<{ kind: string; detail?: string; received?: number; total?: number }>("runtime-progress", (e) => cb(e.payload)),
  onInferenceStatus: (cb: (status: string) => void): Promise<UnlistenFn> =>
    listen<string>("inference-status", (e) => cb(e.payload)),
  onWorkerProgress: (cb: (p: { kind: string; detail?: string }) => void): Promise<UnlistenFn> =>
    listen<{ kind: string; detail?: string }>("worker-progress", (e) => cb(e.payload)),
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
