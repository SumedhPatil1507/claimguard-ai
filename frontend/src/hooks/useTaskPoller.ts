"use client";

import { useCallback, useRef, useState } from "react";
import { pollTask, ApiError } from "@/lib/api";
import type { TaskStatusResponse } from "@/types";

interface UseTaskPollerReturn {
  taskResult: TaskStatusResponse | null;
  taskError: string | null;
  isPolling: boolean;
  startPolling: (taskId: string) => void;
  reset: () => void;
}

export function useTaskPoller(): UseTaskPollerReturn {
  const [taskResult, setTaskResult] = useState<TaskStatusResponse | null>(null);
  const [taskError, setTaskError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const abortRef = useRef(false);

  const startPolling = useCallback((taskId: string) => {
    abortRef.current = false;
    setIsPolling(true);
    setTaskResult(null);
    setTaskError(null);

    pollTask(taskId)
      .then((result) => {
        if (!abortRef.current) {
          setTaskResult(result);
          if (result.status === "FAILURE") {
            setTaskError(result.error ?? "Task failed");
          }
        }
      })
      .catch((err: ApiError) => {
        if (!abortRef.current) setTaskError(err.message);
      })
      .finally(() => {
        if (!abortRef.current) setIsPolling(false);
      });
  }, []);

  const reset = useCallback(() => {
    abortRef.current = true;
    setTaskResult(null);
    setTaskError(null);
    setIsPolling(false);
  }, []);

  return { taskResult, taskError, isPolling, startPolling, reset };
}
