"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL, getToken } from "@/lib/api";
import type { TaskStatusResponse } from "@/types";

export interface TaskProgress {
  stage: string;
  percent: number;
}

export type TaskStatusType =
  | "IDLE"
  | "PENDING"
  | "STARTED"
  | "PROGRESS"
  | "SUCCESS"
  | "FAILURE"
  | "RETRY";

export interface UseTaskStreamReturn {
  taskId: string | null;
  taskResult: TaskStatusResponse | null;
  taskError: string | null;
  progress: TaskProgress | null;
  status: TaskStatusType;
  isStreaming: boolean;
  startStream: (id: string, prefix?: "underwrite" | "claims" | "tasks") => void;
  reset: () => void;
}

export function useTaskStream(
  initialTaskId: string | null = null,
  initialPrefix: "underwrite" | "claims" | "tasks" = "tasks",
): UseTaskStreamReturn {
  const [taskId, setTaskId] = useState<string | null>(initialTaskId);
  const [taskResult, setTaskResult] = useState<TaskStatusResponse | null>(null);
  const [taskError, setTaskError] = useState<string | null>(null);
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [status, setStatus] = useState<TaskStatusType>("IDLE");
  const [isStreaming, setIsStreaming] = useState(false);

  const eventSourceRef = useRef<EventSource | null>(null);

  const closeStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const reset = useCallback(() => {
    closeStream();
    setTaskId(null);
    setTaskResult(null);
    setTaskError(null);
    setProgress(null);
    setStatus("IDLE");
  }, [closeStream]);

  const startStream = useCallback(
    (id: string, prefix: "underwrite" | "claims" | "tasks" = initialPrefix) => {
      closeStream();
      setTaskId(id);
      setTaskResult(null);
      setTaskError(null);
      setStatus("PENDING");
      setProgress({ stage: "Connecting to real-time event stream…", percent: 10 });
      setIsStreaming(true);

      const token = getToken();
      const tokenParam = token ? `?token=${encodeURIComponent(token)}` : "";
      const streamUrl = `${API_BASE_URL}/${prefix}/stream/${id}${tokenParam}`;

      try {
        const es = new EventSource(streamUrl);
        eventSourceRef.current = es;

        es.onmessage = (event) => {
          try {
            const data: TaskStatusResponse = JSON.parse(event.data);
            setTaskResult(data);
            const currentStatus = (data.status as TaskStatusType) || "PENDING";
            setStatus(currentStatus);

            if (data.progress) {
              const p = data.progress as Record<string, unknown>;
              setProgress({
                stage: typeof p.stage === "string" ? p.stage : "Processing…",
                percent: typeof p.percent === "number" ? p.percent : 50,
              });
            } else if (currentStatus === "STARTED") {
              setProgress({ stage: "Task started on worker", percent: 30 });
            } else if (currentStatus === "PROGRESS") {
              setProgress({ stage: "Inference & feature computation in progress", percent: 70 });
            } else if (currentStatus === "SUCCESS") {
              setProgress({ stage: "Completed", percent: 100 });
            } else if (currentStatus === "FAILURE") {
              setProgress({ stage: "Failed", percent: 100 });
            }

            if (currentStatus === "SUCCESS") {
              closeStream();
            } else if (currentStatus === "FAILURE") {
              setTaskError(data.error ?? "Task execution failed.");
              closeStream();
            }
          } catch (err) {
            console.error("Failed to parse SSE payload:", err);
          }
        };

        es.onerror = (err) => {
          console.warn("SSE EventSource stream closed or encountered error:", err);
          if (es.readyState === EventSource.CLOSED) {
            closeStream();
          }
        };
      } catch (err) {
        setTaskError(err instanceof Error ? err.message : "Failed to open EventSource stream");
        closeStream();
      }
    },
    [closeStream, initialPrefix],
  );

  useEffect(() => {
    if (initialTaskId) {
      startStream(initialTaskId, initialPrefix);
    }
    return () => {
      closeStream();
    };
  }, [initialTaskId, initialPrefix, startStream, closeStream]);

  return {
    taskId,
    taskResult,
    taskError,
    progress,
    status,
    isStreaming,
    startStream,
    reset,
  };
}
