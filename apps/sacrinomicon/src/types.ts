export interface Source { id: string; title: string; uri: string; publisher: string; durationSeconds: number; capturedAt: number; summary: string | null }
export interface Provenance { method: string; model: string | null; generatedAt: number; originalAsset: string | null }
export interface Evidence { id: string; sourceId: string; start: number; end: number; text: string; kind: string; provenance: Provenance; vector: number[] }
export interface Claim { id: string; sourceId: string; statement: string; kind: string; evidenceIds: string[]; provenance: Provenance }
export interface Concept { id: string; label: string; aliases: string[]; evidenceIds: string[] }
export interface Relationship { id: string; fromId: string; toId: string; kind: string; evidenceIds: string[]; explanation: string; createdBy: string }
export interface SpellingCorrection { original: string; corrected: string }
export interface Assessment { entityId: string; status: string; note: string; correctedText?: string; sourceSpellings?: SpellingCorrection[]; updatedAt: number }
export interface Asset { id: string; name: string; kind: string; sha256: string; bytes: number }
export interface AnnotationConflict { id: string; incoming: Assessment }
export interface SourceConflict { sourceId: string; revision: string }
export interface ImportReport { sourceIds: string[]; annotationConflicts: number; sourceConflicts: string[] }
export interface RevisionInfo { revision: string; exportedAt: number; evidenceCount: number; active: boolean }
export interface Library { sources: Source[]; evidenceCount: number; concepts: Concept[]; claims: Claim[]; relationships: Relationship[]; assessments: Assessment[]; annotationConflicts: AnnotationConflict[]; sourceConflicts: SourceConflict[] }
export interface SourceDetail { source: Source; evidence: Evidence[]; claims: Claim[]; assessments: Assessment[]; assets: Asset[]; revisions: RevisionInfo[] }
export interface SearchHit { evidence: Evidence; sourceTitle: string; score: number; assessment: Assessment | null; reading?: string | null; sourceSpellings?: SpellingCorrection[] }
export interface ModelStatus { id: string; label: string; bytes: number; downloaded: boolean }
export interface Message { id: string; role: "user" | "assistant"; text: string; sources?: SearchHit[]; streaming?: boolean; error?: boolean }
