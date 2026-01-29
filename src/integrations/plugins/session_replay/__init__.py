"""Session replay integration plugins.

This module provides plugins for session replay and behavior analytics tools:

- LogRocket: Session replay with error tracking and performance monitoring
- FullStory: Digital experience intelligence with session replay
- PostHog: Open-source product analytics with session replay and feature flags
- Hotjar: Heatmaps, session recordings, surveys, and feedback
- Heap: Auto-capture analytics with session replay

Note on API Limitations:
-----------------------
Most session replay tools have limited API access:

1. Hotjar:
   - API is primarily read-only for surveys/feedback
   - Session recordings and heatmaps are NOT accessible via API
   - Must use dashboard to view recordings

2. Heap:
   - Track API (all plans): Write-only for events/users
   - Query API (Enterprise): Export events, users, sessions
   - Session replay videos are dashboard-only

3. LogRocket:
   - Full API access for sessions, issues, and users
   - Session replay URLs are provided for dashboard viewing

4. FullStory:
   - Full API access for sessions, events, users, and segments
   - Replay URLs included in session data

5. PostHog:
   - Full API access for events, persons, recordings, and feature flags
   - Supports both cloud and self-hosted instances

For full session replay capabilities, users should access the
respective dashboards directly. These plugins provide what
integration is possible given API limitations.
"""

from src.integrations.plugins.session_replay.fullstory_plugin import (
    FullStoryEvent,
    FullStoryPlugin,
    FullStorySegment,
    FullStorySession,
    FullStoryUser,
    create_fullstory_plugin,
)
from src.integrations.plugins.session_replay.heap_plugin import (
    HeapPlugin,
    create_heap_plugin,
)
from src.integrations.plugins.session_replay.hotjar_plugin import (
    HotjarPlugin,
    create_hotjar_plugin,
)
from src.integrations.plugins.session_replay.logrocket_plugin import (
    LogRocketIssue,
    LogRocketPlugin,
    LogRocketSession,
    LogRocketUser,
    create_logrocket_plugin,
)
from src.integrations.plugins.session_replay.posthog_plugin import (
    PostHogEvent,
    PostHogFeatureFlag,
    PostHogPerson,
    PostHogPlugin,
    PostHogSessionRecording,
    create_posthog_plugin,
)

__all__ = [
    # LogRocket
    "LogRocketPlugin",
    "LogRocketSession",
    "LogRocketIssue",
    "LogRocketUser",
    "create_logrocket_plugin",
    # FullStory
    "FullStoryPlugin",
    "FullStorySession",
    "FullStoryEvent",
    "FullStoryUser",
    "FullStorySegment",
    "create_fullstory_plugin",
    # PostHog
    "PostHogPlugin",
    "PostHogEvent",
    "PostHogPerson",
    "PostHogSessionRecording",
    "PostHogFeatureFlag",
    "create_posthog_plugin",
    # Hotjar
    "HotjarPlugin",
    "create_hotjar_plugin",
    # Heap
    "HeapPlugin",
    "create_heap_plugin",
]
