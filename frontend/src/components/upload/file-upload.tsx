"use client";

/**
 * FileUpload — Shadcn-style drag-and-drop uploader for the Dual Ingestion
 * Engine.  Native HTML5 drag events (no extra dependencies) with a neon
 * drop-zone, keyboard-accessible browse button, and per-file status chip.
 */

import { useCallback, useRef, useState } from "react";
import { FileSpreadsheet, FileText, Loader2, UploadCloud, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileUploadProps {
  /** Resolve after the upload round-trip completes; reject to surface errors. */
  onUpload: (file: File) => Promise<void>;
  /** Accepted extensions, e.g. ".csv,.pdf,.md" */
  accept?: string;
  disabled?: boolean;
  hint?: string;
}

const DEFAULT_ACCEPT = ".csv,.pdf,.md,.txt,.png,.jpg,.jpeg,.tif,.tiff";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function FileUpload({
  onUpload,
  accept = DEFAULT_ACCEPT,
  disabled = false,
  hint,
}: FileUploadProps) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (next: File | null | undefined) => {
      if (!next || disabled) return;
      setFile(next);
      setError(null);
      setBusy(true);
      try {
        await onUpload(next);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setBusy(false);
      }
    },
    [onUpload, disabled],
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      void handleFile(e.dataTransfer.files?.[0]);
    },
    [handleFile, disabled],
  );

  const onDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (!disabled) setDragging(true);
    },
    [disabled],
  );

  const Icon = file && file.name.toLowerCase().endsWith(".csv") ? FileSpreadsheet : FileText;

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload file"
        onClick={() => !disabled && !busy && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !disabled && !busy) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={() => setDragging(false)}
        className={cn(
          "relative rounded-xl border-2 border-dashed px-6 py-10 text-center cursor-pointer transition-all duration-200",
          "hover:border-primary/60 hover:bg-primary/5 focus:outline-none focus:ring-2 focus:ring-primary/40",
          dragging
            ? "border-primary bg-primary/10 scale-[1.01] shadow-[0_0_24px_rgba(0,229,153,0.15)]"
            : "border-border",
          (disabled || busy) && "pointer-events-none opacity-60",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => void handleFile(e.target.files?.[0])}
        />

        <div className="flex flex-col items-center gap-2">
          {busy ? (
            <Loader2 className="w-9 h-9 text-primary animate-spin" />
          ) : (
            <UploadCloud
              className={cn(
                "w-9 h-9 transition-colors",
                dragging ? "text-primary" : "text-muted-foreground",
              )}
            />
          )}
          <p className="text-sm font-medium">
            {busy ? "Ingesting…" : dragging ? "Release to ingest" : "Drag & drop a file here"}
          </p>
          <p className="text-xs text-muted-foreground">
            or <span className="text-primary underline underline-offset-2">browse</span>
            {" · "}CSV (batch scoring) · PDF/Images (OCR) · Markdown
          </p>
          {hint && <p className="text-[11px] text-muted-foreground/70 mt-1">{hint}</p>}
        </div>

        {/* animated glow while dragging */}
        {dragging && (
          <div className="absolute inset-0 rounded-xl ring-2 ring-primary/40 animate-pulse pointer-events-none" />
        )}
      </div>

      {file && (
        <div className="flex items-center justify-between gap-3 bg-secondary/40 border border-border rounded-lg px-3 py-2">
          <div className="flex items-center gap-2 min-w-0">
            <Icon className="w-4 h-4 text-primary shrink-0" />
            <span className="text-xs font-medium truncate">{file.name}</span>
            <span className="text-[11px] text-muted-foreground shrink-0">
              {formatBytes(file.size)}
            </span>
          </div>
          {!busy && (
            <button
              type="button"
              onClick={() => {
                setFile(null);
                setError(null);
              }}
              className="text-muted-foreground hover:text-foreground transition"
              aria-label="Clear file"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg px-3 py-2 text-xs">
          <X className="w-3.5 h-3.5 shrink-0" />
          {error}
        </div>
      )}
    </div>
  );
}
