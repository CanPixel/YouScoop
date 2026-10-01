import { invoke as nativeInvoke, isTauri } from "@tauri-apps/api/core";
import type { Library, ModelStatus, SourceDetail, Evidence, SearchHit, ImportReport, SpellingCorrection } from "./types";
function invoke<T = unknown>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri()) return Promise.reject(new Error("Open SACRiNOMiCON as a native app to access your local library and models."));
  return nativeInvoke<T>(command, args);
}
export const api = {
  library: () => invoke<Library>("library"),
  source: (id: string) => invoke<SourceDetail>("source_detail", { id }),
  evidence: (id: string) => invoke<Evidence>("evidence_detail", { id }),
  reading: (id: string) => invoke<string>("evidence_reading", { id }),
  assetPath: (sourceId: string, assetId: string) => invoke<string>("asset_path", { sourceId, assetId }),
  assess: (entityId: string, status: string, note: string) => invoke("assess", { entityId, status, note }),
  correctReading: (evidenceId: string, corrected: string | null) => invoke("correct_reading", { evidenceId, corrected }),
  correctSourceSpellings: (sourceId: string, spellings: SpellingCorrection[]) => invoke("correct_source_spellings", { sourceId, spellings }),
  resolveAnnotation: (id: string, useIncoming: boolean) => invoke("resolve_annotation", { id, useIncoming }),
  activateRevision: (sourceId: string, revision: string) => invoke("activate_revision", { sourceId, revision }),
  exportLibrary: () => invoke<{ saved: boolean; path: string | null }>("export_library"),
  addConcept: (label: string, evidenceId: string) => invoke("add_concept", { label, evidenceId }),
  addClaim: (evidenceId: string, statement: string, kind: string) => invoke("add_claim", { evidenceId, statement, kind }),
  addRelationship: (from: string, to: string, kind: string, evidenceId: string, explanation: string) => invoke("add_relationship", { from, to, kind, evidenceId, explanation }),
  search: (query: string, sourceId: string | null = null) => invoke<SearchHit[]>("search", { query, sourceId }),
  modelStatus: () => invoke<ModelStatus[]>("model_status"),
  download: (id: string) => invoke("download_model", { id }),
  unload: () => invoke("unload_models"),
  stop: () => invoke("stop_oracle"),
  voiceMode: (enabled: boolean) => invoke("voice_mode", { enabled }),
  transcribe: (samples: Float32Array) => invoke<string>("transcribe_voice", { samples: Array.from(samples) }),
  speak: (text: string) => invoke("speak_text", { text }),
  stopSpeech: () => invoke("stop_speech"),
  ask: (question: string, sourceId: string | null, history: [string, string][]) => invoke<{ answer: string; sources: SearchHit[] }>("ask_oracle", { question, sourceId, history }),
  import: async (file: File, progress: (fraction: number) => void) => {
    const id = await invoke<string>("begin_import");
    try {
      let offset = 0;
      while (offset < file.size) {
        const bytes = new Uint8Array(await file.slice(offset, offset + 65536).arrayBuffer());
        await invoke("append_import", { id, offset, bytes: Array.from(bytes) });
        offset += bytes.length; progress(offset / file.size);
      }
      return await invoke<ImportReport>("finish_import", { id });
    } catch (error) {
      await invoke("cancel_import", { id }).catch(() => {}); throw error;
    }
  },
};
