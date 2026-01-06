'use client';

import { useChat, Message } from 'ai/react';
import { useRef, useEffect, useState, useCallback } from 'react';
import {
  Send,
  Loader2,
  Bot,
  User,
  Sparkles,
  TestTube,
  Eye,
  Compass,
  CheckCircle,
  XCircle,
  Play,
  Search,
  Code,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface ChatInterfaceProps {
  conversationId?: string;
  initialMessages?: Message[];
  onMessagesChange?: (messages: Message[]) => void;
}

const SUGGESTIONS = [
  { icon: TestTube, text: 'Create a login test for email test@example.com', color: 'text-green-500' },
  { icon: Eye, text: 'Discover all interactive elements on https://demo.vercel.store', color: 'text-blue-500' },
  { icon: Compass, text: 'Run a test: Go to demo.vercel.store, click on a product, add to cart', color: 'text-purple-500' },
  { icon: Sparkles, text: 'Extract all product names from https://demo.vercel.store', color: 'text-orange-500' },
];

// Tool call display component
function ToolCallDisplay({ toolName, args, result, isLoading }: {
  toolName: string;
  args: Record<string, unknown>;
  result?: unknown;
  isLoading?: boolean;
}) {
  const [expanded, setExpanded] = useState(true);
  const [copied, setCopied] = useState(false);

  const toolIcons: Record<string, typeof Play> = {
    executeAction: Play,
    runTest: TestTube,
    discoverElements: Search,
    extractData: Code,
    runAgent: Sparkles,
  };

  const toolLabels: Record<string, string> = {
    executeAction: 'Executing Action',
    runTest: 'Running Test',
    discoverElements: 'Discovering Elements',
    extractData: 'Extracting Data',
    runAgent: 'Running Agent',
  };

  const Icon = toolIcons[toolName] || Play;
  const label = toolLabels[toolName] || toolName;
  const urlValue = typeof args.url === 'string' ? args.url : null;
  const argsJson = JSON.stringify(args, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getStatusIcon = () => {
    if (isLoading) {
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
    }
    if (result && typeof result === 'object' && result !== null && 'success' in result) {
      return (result as { success: boolean }).success
        ? <CheckCircle className="h-4 w-4 text-green-500" />
        : <XCircle className="h-4 w-4 text-red-500" />;
    }
    return <Icon className="h-4 w-4 text-primary" />;
  };

  return (
    <div className="my-3 rounded-lg border bg-muted/30 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center gap-3 hover:bg-muted/50 transition-colors"
      >
        {getStatusIcon()}
        <span className="font-medium text-sm">{label}</span>
        {urlValue && (
          <span className="text-xs text-muted-foreground truncate max-w-[200px]">
            {urlValue}
          </span>
        )}
        <div className="ml-auto">
          {expanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-3 space-y-3">
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">Parameters</div>
                <pre className="text-xs bg-background rounded p-2 overflow-x-auto">
                  {argsJson}
                </pre>
              </div>

              {result !== undefined && result !== null && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-xs font-medium text-muted-foreground">Result</div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleCopy}
                      className="h-6 px-2"
                    >
                      {copied ? (
                        <Check className="h-3 w-3 text-green-500" />
                      ) : (
                        <Copy className="h-3 w-3" />
                      )}
                    </Button>
                  </div>
                  <ResultDisplay result={result} />
                </div>
              )}

              {isLoading && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Processing...</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Screenshot gallery component
function ScreenshotGallery({ screenshots, label }: { screenshots: string[]; label?: string }) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  if (!screenshots || screenshots.length === 0) return null;

  return (
    <div className="mt-3">
      <div className="text-xs font-medium text-muted-foreground mb-2">
        {label || `Screenshots (${screenshots.length})`}
      </div>
      <div className="flex gap-2 overflow-x-auto pb-2">
        {screenshots.map((screenshot, index) => (
          <button
            key={index}
            onClick={() => setSelectedIndex(index)}
            className="relative flex-shrink-0 w-24 h-16 rounded border hover:border-primary transition-colors overflow-hidden"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={screenshot.startsWith('data:') ? screenshot : `data:image/png;base64,${screenshot}`}
              alt={`Step ${index + 1}`}
              className="w-full h-full object-cover"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-[10px] px-1">
              Step {index + 1}
            </div>
          </button>
        ))}
      </div>

      {/* Lightbox modal */}
      <AnimatePresence>
        {selectedIndex !== null && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
            onClick={() => setSelectedIndex(null)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="relative max-w-4xl max-h-[80vh] overflow-hidden rounded-lg"
              onClick={(e) => e.stopPropagation()}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={screenshots[selectedIndex].startsWith('data:')
                  ? screenshots[selectedIndex]
                  : `data:image/png;base64,${screenshots[selectedIndex]}`}
                alt={`Step ${selectedIndex + 1}`}
                className="max-w-full max-h-[80vh] object-contain"
              />
              <div className="absolute top-2 right-2 flex gap-2">
                {selectedIndex > 0 && (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setSelectedIndex(selectedIndex - 1)}
                  >
                    Previous
                  </Button>
                )}
                {selectedIndex < screenshots.length - 1 && (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setSelectedIndex(selectedIndex + 1)}
                  >
                    Next
                  </Button>
                )}
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setSelectedIndex(null)}
                >
                  <XCircle className="h-4 w-4" />
                </Button>
              </div>
              <div className="absolute bottom-2 left-2 bg-black/50 text-white text-sm px-2 py-1 rounded">
                Step {selectedIndex + 1} of {screenshots.length}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Result display component
function ResultDisplay({ result }: { result: unknown }) {
  if (!result || typeof result !== 'object') {
    return (
      <pre className="text-xs bg-background rounded p-2 overflow-x-auto">
        {JSON.stringify(result, null, 2)}
      </pre>
    );
  }

  const data = result as Record<string, unknown>;

  // Extract screenshots from various possible locations
  const screenshots: string[] = [];
  if (Array.isArray(data.screenshots)) {
    screenshots.push(...(data.screenshots as string[]));
  }
  if (typeof data.screenshot === 'string') {
    screenshots.push(data.screenshot);
  }
  if (typeof data.finalScreenshot === 'string') {
    screenshots.push(data.finalScreenshot);
  }

  // Handle actions array (from discoverElements)
  if (Array.isArray(data.actions) && data.actions.length > 0 && !data.steps) {
    return (
      <div className="space-y-2">
        <div className="text-xs text-muted-foreground">
          Found {data.actions.length} interactive elements:
        </div>
        <div className="max-h-60 overflow-y-auto space-y-1">
          {data.actions.slice(0, 20).map((action: { description?: string; selector?: string; method?: string }, index: number) => (
            <div
              key={index}
              className="text-xs bg-background rounded p-2 flex items-start gap-2"
            >
              <span className="text-muted-foreground font-mono">{index + 1}.</span>
              <div>
                <div className="font-medium">{action.description || 'Unknown element'}</div>
                {action.selector && (
                  <code className="text-[10px] text-muted-foreground">{action.selector}</code>
                )}
              </div>
            </div>
          ))}
          {data.actions.length > 20 && (
            <div className="text-xs text-muted-foreground text-center py-2">
              +{data.actions.length - 20} more elements
            </div>
          )}
        </div>
        {screenshots.length > 0 && <ScreenshotGallery screenshots={screenshots} />}
      </div>
    );
  }

  // Handle test steps (from runTest)
  if (Array.isArray(data.steps)) {
    return (
      <div className="space-y-2">
        <div className="text-xs text-muted-foreground">
          Test {data.success ? 'passed' : 'failed'}:
        </div>
        <div className="space-y-1">
          {data.steps.map((step: { instruction?: string; success?: boolean; screenshot?: string }, index: number) => (
            <div
              key={index}
              className={cn(
                "text-xs rounded p-2 flex items-center gap-2",
                step.success ? "bg-green-500/10" : "bg-red-500/10"
              )}
            >
              {step.success ? (
                <CheckCircle className="h-3 w-3 text-green-500" />
              ) : (
                <XCircle className="h-3 w-3 text-red-500" />
              )}
              <span>{step.instruction}</span>
            </div>
          ))}
        </div>
        {screenshots.length > 0 && (
          <ScreenshotGallery screenshots={screenshots} label="Test Screenshots" />
        )}
      </div>
    );
  }

  // Handle single screenshot with message (from executeAction)
  if (data.message && screenshots.length > 0) {
    return (
      <div className="space-y-2">
        <div className="text-xs">
          {data.success ? (
            <span className="text-green-500">✓ {String(data.message)}</span>
          ) : (
            <span className="text-red-500">✗ {String(data.message)}</span>
          )}
        </div>
        <ScreenshotGallery screenshots={screenshots} label="Action Screenshot" />
      </div>
    );
  }

  // Default JSON display with screenshots if available
  return (
    <div>
      <pre className="text-xs bg-background rounded p-2 overflow-x-auto max-h-40">
        {JSON.stringify(result, null, 2)}
      </pre>
      {screenshots.length > 0 && <ScreenshotGallery screenshots={screenshots} />}
    </div>
  );
}

// Message content renderer
function MessageContent({ message }: { message: Message }) {
  // Handle tool invocations (new AI SDK format)
  if (message.toolInvocations && message.toolInvocations.length > 0) {
    return (
      <div>
        {message.content && (
          <div className="prose prose-sm dark:prose-invert max-w-none mb-3">
            {message.content.split('\n').map((line, i) => (
              <p key={i} className="mb-1 last:mb-0">
                {line || '\u00A0'}
              </p>
            ))}
          </div>
        )}
        {message.toolInvocations.map((tool, index) => (
          <ToolCallDisplay
            key={index}
            toolName={tool.toolName}
            args={tool.args as Record<string, unknown>}
            result={tool.state === 'result' ? tool.result : undefined}
            isLoading={tool.state === 'call'}
          />
        ))}
      </div>
    );
  }

  // Regular text content
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      {message.content.split('\n').map((line, i) => (
        <p key={i} className="mb-1 last:mb-0">
          {line || '\u00A0'}
        </p>
      ))}
    </div>
  );
}

export function ChatInterface({ conversationId, initialMessages = [], onMessagesChange }: ChatInterfaceProps) {
  // Track last saved message count to prevent duplicate saves
  const lastSavedCountRef = useRef(0);
  const [error, setError] = useState<string | null>(null);

  // Store conversationId in ref to avoid stale closure issues
  const conversationIdRef = useRef(conversationId);
  const onMessagesChangeRef = useRef(onMessagesChange);

  // Keep refs updated
  useEffect(() => {
    conversationIdRef.current = conversationId;
  }, [conversationId]);

  useEffect(() => {
    onMessagesChangeRef.current = onMessagesChange;
  }, [onMessagesChange]);

  // Validate conversationId is a proper UUID before using
  const isValidConversationId = conversationId &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(conversationId);

  const { messages, input, handleInputChange, handleSubmit, isLoading, setInput, stop } = useChat({
    api: '/api/chat',
    // Only pass ID if it's a valid UUID, otherwise let AI SDK generate one temporarily
    id: isValidConversationId ? conversationId : undefined,
    initialMessages,
    maxSteps: 5, // Allow multi-step tool calls
    onError: (err) => {
      console.error('Chat error:', err);
      setError(err.message || 'An error occurred while processing your request');
    },
    onFinish: (message) => {
      setError(null); // Clear any previous errors on success
      // Use refs to get latest values, avoiding stale closures
      const currentConversationId = conversationIdRef.current;
      const currentOnMessagesChange = onMessagesChangeRef.current;

      // Extra validation: only proceed if we have a valid UUID conversation
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
      if (!currentConversationId || !uuidRegex.test(currentConversationId)) {
        console.warn('ChatInterface: Cannot persist - invalid conversation ID:', currentConversationId);
        return;
      }

      // Persist messages when AI response completes
      if (currentOnMessagesChange && messages.length > lastSavedCountRef.current) {
        currentOnMessagesChange(messages);
        lastSavedCountRef.current = messages.length;
      }
    },
  });

  // Persist when user sends a message (no setTimeout needed)
  const handleSubmitWithPersist = useCallback((e: React.FormEvent<HTMLFormElement>) => {
    handleSubmit(e);
  }, [handleSubmit]);

  // REMOVED: useEffect that was causing infinite re-renders
  // Messages are now only persisted on explicit events (onFinish)

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSuggestionClick = (text: string) => {
    setInput(text);
    inputRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)]">
      {/* Chat Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto space-y-4 p-4"
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full space-y-8">
            <div className="text-center space-y-2">
              <div className="flex justify-center">
                <div className="p-4 rounded-full bg-gradient-to-br from-primary/20 to-purple-500/20">
                  <Bot className="w-12 h-12 text-primary" />
                </div>
              </div>
              <h2 className="text-2xl font-bold">Hey Argus</h2>
              <p className="text-muted-foreground max-w-md">
                Your AI testing companion. I can create tests, discover elements, run browser automations, and extract data.
                Just describe what you want in plain English.
              </p>
            </div>

            {/* Suggestions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl w-full">
              {SUGGESTIONS.map((suggestion, index) => (
                <motion.button
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => handleSuggestionClick(suggestion.text)}
                  className="flex items-center gap-3 p-4 rounded-lg border bg-card hover:bg-accent transition-colors text-left group"
                >
                  <div className={cn(
                    "p-2 rounded-md bg-muted group-hover:bg-primary/10 transition-colors",
                    suggestion.color
                  )}>
                    <suggestion.icon className="w-5 h-5" />
                  </div>
                  <span className="text-sm">{suggestion.text}</span>
                </motion.button>
              ))}
            </div>
          </div>
        ) : (
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className={cn(
                  'flex gap-3',
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                )}
              >
                {message.role === 'assistant' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-primary" />
                  </div>
                )}
                <Card className={cn(
                  'max-w-[85%]',
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-card'
                )}>
                  <CardContent className="p-3">
                    {message.role === 'user' ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        {message.content.split('\n').map((line, i) => (
                          <p key={i} className="mb-1 last:mb-0 text-primary-foreground">
                            {line || '\u00A0'}
                          </p>
                        ))}
                      </div>
                    ) : (
                      <MessageContent message={message} />
                    )}
                  </CardContent>
                </Card>
                {message.role === 'user' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                    <User className="w-5 h-5" />
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        )}

        {/* Loading indicator */}
        {isLoading && messages.length > 0 && messages[messages.length - 1]?.role === 'user' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-3"
          >
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center">
              <Bot className="w-5 h-5 text-primary" />
            </div>
            <Card className="bg-card">
              <CardContent className="p-3">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Processing your request...</span>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mx-4 mb-2 p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm flex items-center justify-between">
          <span>{error}</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setError(null)}
            className="h-6 px-2 text-destructive hover:text-destructive"
          >
            <XCircle className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Input Area */}
      <div className="border-t bg-background/80 backdrop-blur-sm p-4">
        <form onSubmit={handleSubmitWithPersist} className="flex gap-2 max-w-4xl mx-auto">
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              value={input}
              onChange={handleInputChange}
              placeholder="Describe what you want to test or discover..."
              disabled={isLoading}
              className="w-full h-11 px-4 rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          {isLoading ? (
            <Button type="button" variant="outline" onClick={stop}>
              Stop
            </Button>
          ) : (
            <Button type="submit" disabled={!input.trim()}>
              <Send className="w-4 h-4" />
            </Button>
          )}
        </form>
        <p className="text-xs text-muted-foreground text-center mt-2">
          Try: "Discover elements on https://demo.vercel.store" or "Run a login test on my app"
        </p>
      </div>
    </div>
  );
}
