'use client';

import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Play,
  Monitor,
  Clock,
  AlertTriangle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Test } from '@/lib/supabase/types';

interface StepResult {
  instruction: string;
  success: boolean;
  error?: string;
  screenshot?: string;
  duration?: number;
}

interface ExecutionState {
  status: 'idle' | 'running' | 'completed' | 'failed';
  currentStep: number;
  steps: StepResult[];
  screenshot?: string;
  error?: string;
  startTime?: number;
  endTime?: number;
}

const WORKER_URL = process.env.NEXT_PUBLIC_E2E_WORKER_URL || 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev';

interface LiveExecutionModalProps {
  test: Test | null;
  appUrl: string;
  open: boolean;
  onClose: () => void;
  onComplete: (success: boolean, results: StepResult[]) => void;
}

export function LiveExecutionModal({
  test,
  appUrl,
  open,
  onClose,
  onComplete,
}: LiveExecutionModalProps) {
  const [execution, setExecution] = useState<ExecutionState>({
    status: 'idle',
    currentStep: 0,
    steps: [],
  });

  const steps = test?.steps
    ? (test.steps as { instruction: string }[]).map((s) => s.instruction)
    : [];

  // Reset state when modal opens with new test
  useEffect(() => {
    if (open && test) {
      setExecution({
        status: 'idle',
        currentStep: 0,
        steps: steps.map((instruction) => ({
          instruction,
          success: false,
        })),
      });
    }
  }, [open, test?.id]);

  const runTest = async () => {
    if (!test || steps.length === 0) return;

    setExecution((prev) => ({
      ...prev,
      status: 'running',
      currentStep: 0,
      startTime: Date.now(),
    }));

    try {
      // Execute test via Worker with extended timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

      const response = await fetch(`${WORKER_URL}/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: appUrl,
          steps,
          browser: 'chrome',
          screenshot: true,
          verbose: true,
          timeout: 45000, // Give worker 45s per step
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const result = await response.json();

      // Process results
      const stepResults: StepResult[] = result.steps?.map((step: any, index: number) => ({
        instruction: steps[index] || step.instruction,
        success: step.success,
        error: step.error,
        screenshot: step.screenshot,
      })) || [];

      // Update with final results
      setExecution({
        status: result.success ? 'completed' : 'failed',
        currentStep: steps.length,
        steps: stepResults,
        screenshot: result.browsers?.[0]?.screenshot,
        error: result.error,
        startTime: execution.startTime,
        endTime: Date.now(),
      });

      onComplete(result.success, stepResults);
    } catch (error) {
      let errorMessage = 'Unknown error';
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Test timed out after 60 seconds. The page may be slow or elements not found.';
        } else {
          errorMessage = error.message;
        }
      }

      setExecution((prev) => ({
        ...prev,
        status: 'failed',
        error: errorMessage,
        endTime: Date.now(),
      }));
      onComplete(false, []);
    }
  };

  const duration = execution.endTime && execution.startTime
    ? ((execution.endTime - execution.startTime) / 1000).toFixed(1)
    : null;

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <Monitor className="h-5 w-5" />
            {test?.name || 'Test Execution'}
            {execution.status === 'running' && (
              <span className="text-sm font-normal text-muted-foreground animate-pulse">
                Running...
              </span>
            )}
            {execution.status === 'completed' && (
              <span className="text-sm font-normal text-success flex items-center gap-1">
                <CheckCircle2 className="h-4 w-4" /> Passed
              </span>
            )}
            {execution.status === 'failed' && (
              <span className="text-sm font-normal text-error flex items-center gap-1">
                <XCircle className="h-4 w-4" /> Failed
              </span>
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-auto">
          <div className="grid grid-cols-2 gap-6">
            {/* Left: Steps */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-sm">Test Steps</h3>
                {duration && (
                  <span className="text-xs text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" /> {duration}s
                  </span>
                )}
              </div>

              <div className="space-y-2">
                {execution.steps.map((step, index) => {
                  const isCurrentStep = index === execution.currentStep && execution.status === 'running';
                  const isCompleted = index < execution.currentStep || execution.status === 'completed' || execution.status === 'failed';

                  return (
                    <div
                      key={index}
                      className={cn(
                        'p-3 rounded-lg border transition-all',
                        isCurrentStep && 'border-primary bg-primary/5 ring-2 ring-primary/20',
                        isCompleted && step.success && 'border-success/30 bg-success/5',
                        isCompleted && !step.success && 'border-error/30 bg-error/5',
                        !isCurrentStep && !isCompleted && 'border-border bg-muted/30'
                      )}
                    >
                      <div className="flex items-start gap-3">
                        <div className="mt-0.5">
                          {isCurrentStep ? (
                            <Loader2 className="h-4 w-4 text-primary animate-spin" />
                          ) : isCompleted && step.success ? (
                            <CheckCircle2 className="h-4 w-4 text-success" />
                          ) : isCompleted && !step.success ? (
                            <XCircle className="h-4 w-4 text-error" />
                          ) : (
                            <div className="h-4 w-4 rounded-full border-2 border-muted-foreground/30" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-medium text-muted-foreground">
                              Step {index + 1}
                            </span>
                          </div>
                          <p className="text-sm mt-1">{step.instruction}</p>
                          {step.error && (
                            <p className="text-xs text-error mt-1 flex items-center gap-1">
                              <AlertTriangle className="h-3 w-3" />
                              {step.error}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {execution.error && (
                <div className="p-3 rounded-lg border border-error/30 bg-error/5">
                  <p className="text-sm text-error font-medium mb-2">Error</p>
                  <p className="text-sm text-error">{execution.error}</p>
                  <div className="mt-3 text-xs text-muted-foreground">
                    <p className="font-medium mb-1">Troubleshooting tips:</p>
                    <ul className="list-disc list-inside space-y-1">
                      <li>Check if the App URL is correct</li>
                      <li>Verify the page has the elements you're targeting</li>
                      <li>Use simple natural language (e.g., "Click the login button")</li>
                      <li>Try testing on https://example.com first</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>

            {/* Right: Screenshot/Preview */}
            <div className="space-y-4">
              <h3 className="font-medium text-sm">Live Preview</h3>

              <div className="aspect-video rounded-lg border bg-muted/50 overflow-hidden relative">
                {execution.screenshot ? (
                  <img
                    src={`data:image/png;base64,${execution.screenshot}`}
                    alt="Test screenshot"
                    className="w-full h-full object-contain"
                  />
                ) : execution.status === 'running' ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                    <p className="text-sm text-muted-foreground">Executing test...</p>
                    <p className="text-xs text-muted-foreground">
                      Step {execution.currentStep + 1} of {steps.length}
                    </p>
                  </div>
                ) : execution.status === 'idle' ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                    <Monitor className="h-8 w-8 text-muted-foreground/50" />
                    <p className="text-sm text-muted-foreground">
                      Click "Run Test" to start
                    </p>
                  </div>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                    {execution.status === 'completed' ? (
                      <CheckCircle2 className="h-8 w-8 text-success" />
                    ) : (
                      <XCircle className="h-8 w-8 text-error" />
                    )}
                    <p className="text-sm text-muted-foreground">
                      {execution.status === 'completed' ? 'Test passed!' : 'Test failed'}
                    </p>
                  </div>
                )}
              </div>

              <div className="text-xs text-muted-foreground">
                <p><strong>URL:</strong> {appUrl}</p>
                <p><strong>Browser:</strong> Chrome (Cloudflare)</p>
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 pt-4 border-t mt-4">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button
            onClick={runTest}
            disabled={execution.status === 'running' || !test}
          >
            {execution.status === 'running' ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Running...
              </>
            ) : execution.status === 'completed' || execution.status === 'failed' ? (
              <>
                <Play className="h-4 w-4 mr-2" />
                Run Again
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Run Test
              </>
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
