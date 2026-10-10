"use client";

/**
 * Data Ingestion — drag-and-drop uploads for the Dual Ingestion Engine.
 *
 * * CSV        → Celery batch enqueue + live SSE progress (EventSource)
 * * PDF/Image  → synchronous OCR + H1–H3 chunking + Qdrant hybrid upsert
 */

import { useCallback, useState } from "react";
import {
  CheckCircle2,
  Database,
  FileStack,
  Radio,
  Sparkles,
  UploadCloud,
  XCircle,
} from "lucide-react";
import { ingestFile, type IngestFileResponse } from "@/lib/api";
import { useTaskStream } from "@/hooks/useTaskStream";
import { FileUpload } from "@/components/upload/file-upload";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ExplainableDecision {
  verdict: string;
  reasoning: string;
  recommendation: string;
  next_steps: string[];
}

interface BatchSummary {
  filename: string;
  kind: string;
  rows_received: number;
  processed: number;
  flagged: number;
  errors: number;
  decision?: ExplainableDecision;
}

interface UploadHistoryItem {
  id: number;
  filename: string;
  status: string;
  detail: string;
}

let historyId = 0;

export default function IngestPage() {
  const [upload, setUpload] = useState<IngestFileResponse | null>(null);
  const [history, setHistory] = useState<UploadHistoryItem[]>([]);
  const { taskResult, taskError, progress, isStreaming, startStream, reset } =
    useTaskStream();

  const pushHistory = useCallback((filename: string, status: string, detail: string) => {
    setHistory((h) =>
      [{ id: ++historyId, filename, status, detail }, ...h].slice(0, 8),
    );
  }, []);

  const handleUpload = useCallback(
    async (file: File) => {
      reset();
      setUpload(null);
      try {
        const resp = await ingestFile(file);
        setUpload(resp);

        if (resp.status === "ENQUEUED" && resp.task_id) {
          // CSV batch → stream live worker progress over SSE (no polling)
          startStream(resp.task_id, "tasks");
          pushHistory(resp.filename, "QUEUED", `${resp.rows} rows → ${resp.dataset} batch`);
        } else {
          pushHistory(
            resp.filename,
            resp.status,
            `${resp.chunks ?? 0} chunks → ${resp.vector_store ?? "vector store"}`,
          );
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Upload failed";
        pushHistory(file.name, "FAILED", msg);
        throw err;
      }
    },
    [pushHistory, reset, startStream],
  );

  const batch = taskResult?.result as BatchSummary | undefined;
  const decision = batch?.decision;
  const docStats = upload && upload.status !== "ENQUEUED" ? upload : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <UploadCloud className="w-6 h-6 text-primary" /> Data Ingestion
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Dual ingestion engine · CSV → Celery batch scoring · PDF/Images → OCR →
          Markdown H1–H3 chunking → Qdrant hybrid store
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* ── Uploader ── */}
        <div className="bg-card border border-border rounded-xl p-6 space-y-4">
          <h2 className="text-sm font-semibold">Upload Files</h2>
          <FileUpload
            onUpload={handleUpload}
            hint="Max 25 MB · CSV up to 10,000 rows per batch"
          />

          {upload && (
            <div className="rounded-lg border border-border bg-secondary/30 px-3 py-2.5 text-xs space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-medium truncate">{upload.filename}</span>
                <Badge>{upload.status}</Badge>
              </div>
              {upload.message && (
                <p className="text-muted-foreground">{upload.message}</p>
              )}
              {upload.task_id && (
                <p className="font-mono text-[10px] text-muted-foreground truncate">
                  task: {upload.task_id}
                </p>
              )}
            </div>
          )}

          {/* Upload history */}
          {history.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                Recent uploads
              </p>
              {history.map((item) => (
                <div
                  key={item.id}
                  className="flex items-center justify-between gap-2 text-xs bg-secondary/30 border border-border/60 rounded-md px-2.5 py-1.5"
                >
                  <span className="truncate font-medium">{item.filename}</span>
                  <span
                    className={cn(
                      "shrink-0 text-[10px] font-semibold px-1.5 py-0.5 rounded-full border",
                      item.status === "FAILED"
                        ? "text-red-400 border-red-400/30 bg-red-400/10"
                        : item.status === "QUEUED"
                          ? "text-amber-400 border-amber-400/30 bg-amber-400/10"
                          : "text-emerald-400 border-emerald-400/30 bg-emerald-400/10",
                    )}
                  >
                    {item.status}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Live progress / results ── */}
        <div className="bg-card border border-border rounded-xl p-6 flex flex-col">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-sm font-semibold flex items-center gap-2">
              Ingestion Status
              {isStreaming && (
                <span className="flex items-center gap-1 text-[11px] font-normal text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
                  <Radio className="w-3 h-3 animate-pulse text-emerald-400" /> SSE Live
                </span>
              )}
            </h2>
            {taskResult && (
              <button
                onClick={reset}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Reset
              </button>
            )}
          </div>

          {/* Idle */}
          {!isStreaming && !taskResult && !docStats && !taskError && (
            <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground text-sm gap-3">
              <FileStack className="w-8 h-8 opacity-40" />
              Drop a file to start ingestion
            </div>
          )}

          {/* CSV batch — live SSE progress bar */}
          {isStreaming && (
            <div className="flex-1 flex flex-col items-center justify-center gap-5 p-6 text-center">
              <div className="relative">
                <div className="w-16 h-16 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
                <Sparkles className="w-6 h-6 text-primary absolute inset-0 m-auto animate-pulse" />
              </div>
              <div className="w-full max-w-sm space-y-2">
                <div className="flex justify-between text-xs text-muted-foreground font-medium">
                  <span className="truncate">{progress?.stage ?? "Queuing batch…"}</span>
                  <span className="font-mono text-primary">{progress?.percent ?? 10}%</span>
                </div>
                <div className="w-full bg-secondary/80 h-2.5 rounded-full overflow-hidden border border-border/50">
                  <div
                    className="bg-primary h-full rounded-full transition-all duration-500 ease-out relative"
                    style={{ width: `${Math.max(5, progress?.percent ?? 10)}%` }}
                  >
                    <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite]" />
                  </div>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                Real-time worker progress over Server-Sent Events (no polling)
              </p>
            </div>
          )}

          {/* Error */}
          {taskError && (
            <div className="flex items-center gap-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg px-4 py-3 text-sm">
              <XCircle className="w-4 h-4 shrink-0" /> {taskError}
            </div>
          )}
          {/* CSV batch result + standardized decision JSON */}
          {!isStreaming && batch && !taskError && (
            <div className="space-y-4 animate-fade-in">
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: "Processed", value: batch.processed },
                  { label: "Flagged", value: batch.flagged },
                  { label: "Errors", value: batch.errors },
                ].map((s) => (
                  <div
                    key={s.label}
                    className="bg-secondary/40 border border-border rounded-lg p-3 text-center"
                  >
                    <p className="text-xl font-bold">{s.value}</p>
                    <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      {s.label}
                    </p>
                  </div>
                ))}
              </div>

              {decision && (
                <div className="rounded-lg border border-primary/30 bg-primary/5 p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-primary" />
                    <span className="text-sm font-bold text-primary">
                      {decision.verdict}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {decision.reasoning}
                  </p>
                  <p className="text-xs">
                    <span className="text-muted-foreground">Recommendation:</span>{" "}
                    <span className="font-semibold">{decision.recommendation}</span>
                  </p>
                  <ul className="text-xs space-y-1 list-disc list-inside text-muted-foreground">
                    {decision.next_steps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          {/* PDF, image, and text ingestion result */}
          {!isStreaming && docStats && !taskError && (
            <div className="flex-1 flex flex-col justify-center gap-4 animate-fade-in">
              <div className="flex items-center gap-3 rounded-lg border border-emerald-500/25 bg-emerald-500/5 p-4">
                <CheckCircle2 className="h-8 w-8 shrink-0 text-emerald-400" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold">{docStats.filename}</p>
                  <p className="text-xs text-muted-foreground">Document processed and indexed</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg border border-border bg-secondary/30 p-4">
                  <p className="text-2xl font-bold">{docStats.chunks ?? 0}</p>
                  <p className="mt-1 text-[10px] uppercase tracking-wide text-muted-foreground">Markdown chunks</p>
                </div>
                <div className="rounded-lg border border-border bg-secondary/30 p-4">
                  <p className="flex items-center gap-2 text-sm font-semibold">
                    <Database className="h-4 w-4 text-primary" /> {docStats.vector_store ?? "Qdrant"}
                  </p>
                  <p className="mt-1 text-[10px] uppercase tracking-wide text-muted-foreground">Hybrid index</p>
                </div>
              </div>
              {docStats.message && (
                <p className="text-xs leading-relaxed text-muted-foreground">{docStats.message}</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
