# Context Management Strategy for LangGraph Agents

## The Problem

E2E testing agents face a unique challenge: they generate **massive amounts of data** (screenshots, HTML snapshots, detailed logs) that are essential for debugging but toxic to LLM context windows.

```
Token breakdown for a typical test run:
├── System prompt:           ~1,000 tokens
├── User messages:           ~500 tokens
├── AI responses:            ~2,000 tokens
├── Tool calls:              ~500 tokens
└── Tool results:            ~150,000 tokens (!!)
    ├── Screenshot 1:        ~50,000 tokens (base64)
    ├── Screenshot 2:        ~50,000 tokens
    └── Step details:        ~50,000 tokens

Total: ~154,000 tokens per test run
After 2 runs: 308,000 tokens > 200k limit 💥
```

## Design Strategies

### Strategy 1: Artifact Store (Recommended) ⭐

**Separate storage for large artifacts, references in state.**

```
┌─────────────────────────────────────────────────────────────────┐
│                     ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Tool Execution                                                │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────┐                                           │
│   │  Full Result    │                                           │
│   │  + Screenshots  │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│     ┌──────┴──────┐                                             │
│     │             │                                              │
│     ▼             ▼                                              │
│ ┌────────┐   ┌────────────┐                                     │
│ │ Stream │   │ Artifact   │                                     │
│ │ Full   │   │ Store      │                                     │
│ │ Data   │   │ (Supabase) │                                     │
│ └───┬────┘   └─────┬──────┘                                     │
│     │              │                                             │
│     │              ▼                                             │
│     │        ┌───────────┐                                      │
│     │        │ Reference │                                      │
│     │        │ Only      │                                      │
│     │        └─────┬─────┘                                      │
│     │              │                                             │
│     ▼              ▼                                             │
│ ┌────────┐   ┌────────────┐                                     │
│ │Frontend│   │ LangGraph  │                                     │
│ │ (full) │   │ State      │                                     │
│ └────────┘   │ (light)    │                                     │
│              └────────────┘                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# In tool_executor_node
from src.orchestrator.artifact_store import get_artifact_store

async def tool_executor_node(state: ChatState, config) -> dict:
    artifact_store = get_artifact_store()

    # Execute tool, get full result
    full_result = await execute_tool(tool_call)

    # Store artifacts, get lightweight result
    lightweight_result = artifact_store.store_test_result(full_result)

    # Stream full result to frontend (before storing)
    await stream_to_frontend(full_result)

    # Store only lightweight in state
    return {"messages": [ToolMessage(content=json.dumps(lightweight_result))]}
```

**Pros:**
- Full data preserved for frontend/debugging
- State stays small and fast
- Claude gets summary + can request details
- Works with existing LangGraph checkpointer

**Cons:**
- Additional storage infrastructure
- Need to manage artifact lifecycle

---

### Strategy 2: Hierarchical Memory

**Three-tier memory system:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TIER 1: Working Memory (in state)                              │
│  ├── Last 5 messages (full detail)                              │
│  ├── Current test context                                       │
│  └── Active tool calls                                          │
│                                                                  │
│  TIER 2: Session Memory (PostgresStore)                         │
│  ├── Summarized older messages                                  │
│  ├── Test results with artifact refs                            │
│  └── Learned patterns from this session                         │
│                                                                  │
│  TIER 3: Long-term Memory (Vector Store)                        │
│  ├── Similar past failures (semantic search)                    │
│  ├── Successful healing strategies                              │
│  └── Application-specific knowledge                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class HierarchicalMemory:
    def __init__(self):
        self.working_memory = []  # Recent messages
        self.session_store = PostgresStore()  # Session summaries
        self.vector_store = PgVector()  # Semantic search

    def add_message(self, message: BaseMessage):
        self.working_memory.append(message)

        # If working memory exceeds limit, summarize and move to session
        if len(self.working_memory) > 10:
            summary = self._summarize(self.working_memory[:5])
            self.session_store.put(summary)
            self.working_memory = self.working_memory[5:]

    def get_context(self, query: str) -> List[BaseMessage]:
        """Get relevant context for current query."""
        context = []

        # Always include working memory
        context.extend(self.working_memory)

        # Add relevant session summaries
        relevant = self.session_store.search(query, k=3)
        context.extend(relevant)

        # Add similar past experiences
        similar = self.vector_store.similarity_search(query, k=2)
        context.extend(similar)

        return context
```

---

### Strategy 3: Conversation Compaction

**Periodically summarize and compact conversation:**

```python
async def compact_conversation_node(state: ChatState, config) -> dict:
    """Periodically compact old messages into summaries."""
    messages = state["messages"]

    # If under threshold, no compaction needed
    estimated_tokens = sum(estimate_tokens(m) for m in messages)
    if estimated_tokens < 100000:
        return {"messages": messages}

    # Keep recent messages
    recent = messages[-10:]
    old = messages[:-10]

    # Summarize old messages using a smaller model
    summary = await summarize_with_haiku(old)

    # Create summary message
    summary_message = SystemMessage(
        content=f"[Previous conversation summary]\n{summary}"
    )

    return {"messages": [summary_message] + recent}
```

---

### Strategy 4: Tool-Specific State (Best for Testing)

**Separate state fields for different data types:**

```python
class TestingState(TypedDict):
    # Conversation (kept small)
    messages: Annotated[List[BaseMessage], add_messages]

    # Test execution (separate from messages)
    current_test: Optional[dict]  # Current test being executed
    test_history: List[dict]  # Summaries of past tests

    # Artifacts (references only)
    screenshot_refs: List[str]  # IDs to artifact store

    # Learning (semantic memory)
    failure_patterns: List[dict]  # Known failure patterns
    healing_strategies: List[dict]  # Successful fixes

    # Context (application-specific)
    app_url: str
    discovered_elements: List[dict]  # Page elements cache
```

**Benefits:**
- Messages stay conversational and small
- Test data organized separately
- Easy to query/filter specific data types
- LangGraph can checkpoint each field independently

---

## Recommended Architecture for Argus

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED DESIGN                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  LangGraph State                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │  messages   │  │ test_state  │  │  artifacts  │     │    │
│  │  │  (conv.)    │  │ (current)   │  │  (refs)     │     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                         │                                        │
│          ┌──────────────┼──────────────┐                        │
│          ▼              ▼              ▼                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ PostgreSQL  │  │  Supabase   │  │  pgvector   │             │
│  │ Checkpointer│  │  Storage    │  │  Memory     │             │
│  │ (state)     │  │ (artifacts) │  │ (patterns)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Flow:
1. Tool executes → Full result with screenshots
2. Stream full result to frontend immediately
3. Extract artifacts → Store in Supabase Storage
4. Create lightweight result with refs
5. Store lightweight in LangGraph state
6. Prune old messages if needed
7. Claude sees summaries + can request artifact details
```

## Implementation Priority

1. **Phase 1 (Current)**: Message pruning + base64 stripping
   - Quick fix, already implemented
   - Handles immediate overflow issue

2. **Phase 2**: Artifact Store
   - Implement `artifact_store.py`
   - Store screenshots in Supabase Storage
   - Pass refs instead of base64 in state

3. **Phase 3**: Hierarchical Memory
   - Add session summaries
   - Implement vector search for similar failures
   - Enable cross-session learning

4. **Phase 4**: Tool-Specific State
   - Refactor state to separate concerns
   - Add test history tracking
   - Implement element caching

## Code Changes Required

### chat_graph.py
```python
# Add artifact store integration
from src.orchestrator.artifact_store import get_artifact_store

async def tool_executor_node(state: ChatState, config) -> dict:
    artifact_store = get_artifact_store()

    for tool_call in last_message.tool_calls:
        # Execute tool
        result = await execute_tool(tool_call)

        # Extract and store artifacts
        lightweight_result = artifact_store.store_test_result(result)

        # Create tool message with lightweight result
        tool_results.append(ToolMessage(
            content=json.dumps(lightweight_result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_results}
```

### chat.py (streaming)
```python
# Stream full artifacts before storing lightweight
async def generate_ai_sdk_stream():
    artifact_store = get_artifact_store()

    async for event in app.astream(...):
        if event_type == "values":
            # Get the artifact refs from lightweight result
            result = last_msg.content
            artifact_refs = result.get("_artifact_refs", [])

            # Stream full artifacts to frontend
            for ref in artifact_refs:
                full_artifact = artifact_store.get(ref["artifact_id"])
                yield f'b:{json.dumps({"type": "artifact", "data": full_artifact})}\n'
```

## Summary

| Strategy | Complexity | Effectiveness | Recommended For |
|----------|------------|---------------|-----------------|
| Message Pruning | Low | Medium | Quick fix |
| Artifact Store | Medium | High | Production |
| Hierarchical Memory | High | Very High | Enterprise |
| Tool-Specific State | Medium | High | Complex agents |

**For Argus, implement in this order:**
1. ✅ Message pruning (done)
2. 🔜 Artifact store (next)
3. 📅 Hierarchical memory (future)
4. 📅 Tool-specific state (future)
