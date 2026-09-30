export interface VideoMeta {
  id: string;
  url: string;
  title: string;
  channel: string;
  durationSecs: number;
  addedAt: number;
  status: "pending" | "processing" | "ready" | "error" | "cancelled";
  error?: string | null;
  videoPath?: string | null;
  thumbPath?: string | null;
  hasCaptions: boolean;
  frameCount: number;
  chunkCount: number;
}

export interface ChunkPublic {
  start: number;
  end: number;
  text: string;
  source: "speech" | "vision";
}

export interface TranscriptCue {
  start: number;
  end: number;
  text: string;
}

export interface VideoDetail {
  meta: VideoMeta;
  chunks: ChunkPublic[];
  transcript: TranscriptCue[];
}

export interface SourceRef {
  start: number;
  end: number;
  source: "speech" | "vision";
  text: string;
}

/** Payload of the `ask_video` command (Rust `ChatAnswer`). */
export interface ChatAnswer {
  answer: string;
  sources: SourceRef[];
}

export interface ModelStatus {
  id: string;
  label: string;
  detail: string;
  bytes: number;
  downloaded: boolean;
  partialBytes: number;
}

export interface SetupStatus {
  models: ModelStatus[];
  ffmpegOk: boolean;
  engineLoaded: boolean;
  engineLoading: boolean;
}

export interface ModelProgress {
  id: string;
  received: number;
  total: number;
  done: boolean;
  error?: string | null;
}

export interface EngineStatus {
  loaded: boolean;
  loading: boolean;
  error?: string | null;
}

export type IngestStage =
  | "metadata"
  | "download"
  | "captions"
  | "transcribe"
  | "frames"
  | "describe"
  | "index"
  | "ready"
  | "error"
  | "cancelled";

export interface IngestProgress {
  videoId: string;
  stage: IngestStage;
  progress: number;
  detail: string;
  meta?: VideoMeta;
}

export interface ChatToken {
  videoId: string;
  token: string;
}

export interface ChatDone {
  videoId: string;
  answer: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
  error?: boolean;
}
