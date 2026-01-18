"""JavaScript snippet generator for rrweb recording.

Generates embeddable code snippets that websites can use to record
user sessions for test generation.
"""

from dataclasses import dataclass


@dataclass
class RecorderConfig:
    """Configuration for the recorder snippet."""

    # Recording options
    block_class: str = "rr-block"
    block_selector: str | None = None
    ignore_class: str = "rr-ignore"
    mask_text_class: str = "rr-mask"
    mask_text_selector: str = "input[type=password]"
    mask_all_inputs: bool = False

    # Upload options
    upload_endpoint: str = "/api/v1/recording/upload"
    upload_interval_ms: int = 10000  # Send events every 10 seconds
    max_events_per_upload: int = 1000

    # Privacy options
    mask_inputs: bool = True
    mask_input_fn: str | None = None

    # Session options
    auto_start: bool = True
    sampling: dict = None

    def __post_init__(self):
        if self.sampling is None:
            self.sampling = {
                "mousemove": True,
                "mouseInteraction": True,
                "scroll": 150,  # Sample scroll events at 150ms intervals
                "input": "last",  # Only record final input value
            }


class RecorderSnippetGenerator:
    """Generates JavaScript snippets for rrweb recording."""

    RRWEB_CDN = "https://cdn.jsdelivr.net/npm/rrweb@2.0.0-alpha.11/dist/rrweb.min.js"
    RRWEB_CSS_CDN = "https://cdn.jsdelivr.net/npm/rrweb@2.0.0-alpha.11/dist/rrweb.min.css"

    def __init__(self, config: RecorderConfig | None = None):
        """Initialize generator with configuration.

        Args:
            config: Recorder configuration
        """
        self.config = config or RecorderConfig()

    def generate_inline_snippet(self, api_key: str = "", project_id: str = "") -> str:
        """Generate inline JavaScript snippet for recording.

        Args:
            api_key: API key for authentication
            project_id: Project ID for organizing recordings

        Returns:
            JavaScript code to embed in page
        """
        config = self.config

        return f'''<script>
(function() {{
  // Argus E2E Recording Snippet
  var ARGUS_API_KEY = "{api_key}";
  var ARGUS_PROJECT_ID = "{project_id}";
  var UPLOAD_ENDPOINT = "{config.upload_endpoint}";

  // Load rrweb
  var script = document.createElement("script");
  script.src = "{self.RRWEB_CDN}";
  script.onload = initRecording;
  document.head.appendChild(script);

  var events = [];
  var sessionId = "rec_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);

  function initRecording() {{
    if (typeof rrweb === "undefined") return;

    rrweb.record({{
      emit: function(event) {{
        events.push(event);
        if (events.length >= {config.max_events_per_upload}) {{
          uploadEvents();
        }}
      }},
      blockClass: "{config.block_class}",
      ignoreClass: "{config.ignore_class}",
      maskTextClass: "{config.mask_text_class}",
      maskTextSelector: "{config.mask_text_selector}",
      maskAllInputs: {str(config.mask_all_inputs).lower()},
      sampling: {{
        mousemove: {str(config.sampling.get("mousemove", True)).lower()},
        mouseInteraction: {str(config.sampling.get("mouseInteraction", True)).lower()},
        scroll: {config.sampling.get("scroll", 150)},
        input: "{config.sampling.get("input", "last")}"
      }}
    }});

    // Upload events periodically
    setInterval(uploadEvents, {config.upload_interval_ms});

    // Upload remaining events before page unload
    window.addEventListener("beforeunload", function() {{
      uploadEvents(true);
    }});

    console.log("[Argus] Recording started:", sessionId);
  }}

  function uploadEvents(sync) {{
    if (events.length === 0) return;

    var payload = {{
      session_id: sessionId,
      project_id: ARGUS_PROJECT_ID,
      events: events.splice(0, events.length),
      metadata: {{
        href: window.location.href,
        title: document.title,
        width: window.innerWidth,
        height: window.innerHeight,
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString()
      }}
    }};

    if (sync && navigator.sendBeacon) {{
      navigator.sendBeacon(UPLOAD_ENDPOINT, JSON.stringify(payload));
    }} else {{
      fetch(UPLOAD_ENDPOINT, {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json",
          "X-Argus-API-Key": ARGUS_API_KEY
        }},
        body: JSON.stringify(payload),
        keepalive: true
      }}).catch(function(err) {{
        console.error("[Argus] Upload failed:", err);
      }});
    }}
  }}
}})();
</script>'''

    def generate_npm_snippet(self, api_key: str = "", project_id: str = "") -> str:
        """Generate NPM/ES module snippet for React/Vue/etc apps.

        Args:
            api_key: API key for authentication
            project_id: Project ID for organizing recordings

        Returns:
            TypeScript/JavaScript module code
        """
        config = self.config

        return f'''// Argus E2E Recording - NPM Module
// Install: npm install rrweb

import {{ record }} from 'rrweb';

const ARGUS_CONFIG = {{
  apiKey: "{api_key}",
  projectId: "{project_id}",
  uploadEndpoint: "{config.upload_endpoint}",
  uploadIntervalMs: {config.upload_interval_ms},
  maxEventsPerUpload: {config.max_events_per_upload}
}};

let events: any[] = [];
let sessionId: string;

export function initArgusRecording() {{
  sessionId = `rec_${{Date.now()}}_${{Math.random().toString(36).substr(2, 9)}}`;

  const stopFn = record({{
    emit(event) {{
      events.push(event);
      if (events.length >= ARGUS_CONFIG.maxEventsPerUpload) {{
        uploadEvents();
      }}
    }},
    blockClass: "{config.block_class}",
    ignoreClass: "{config.ignore_class}",
    maskTextClass: "{config.mask_text_class}",
    maskTextSelector: "{config.mask_text_selector}",
    maskAllInputs: {str(config.mask_all_inputs).lower()},
    sampling: {{
      mousemove: {str(config.sampling.get("mousemove", True)).lower()},
      mouseInteraction: {str(config.sampling.get("mouseInteraction", True)).lower()},
      scroll: {config.sampling.get("scroll", 150)},
      input: "{config.sampling.get("input", "last")}"
    }}
  }});

  // Upload periodically
  const intervalId = setInterval(uploadEvents, ARGUS_CONFIG.uploadIntervalMs);

  // Upload on unload
  window.addEventListener("beforeunload", () => uploadEvents(true));

  console.log("[Argus] Recording started:", sessionId);

  return () => {{
    stopFn();
    clearInterval(intervalId);
    uploadEvents(true);
  }};
}}

async function uploadEvents(sync = false) {{
  if (events.length === 0) return;

  const payload = {{
    session_id: sessionId,
    project_id: ARGUS_CONFIG.projectId,
    events: events.splice(0, events.length),
    metadata: {{
      href: window.location.href,
      title: document.title,
      width: window.innerWidth,
      height: window.innerHeight,
      userAgent: navigator.userAgent,
      timestamp: new Date().toISOString()
    }}
  }};

  if (sync && navigator.sendBeacon) {{
    navigator.sendBeacon(ARGUS_CONFIG.uploadEndpoint, JSON.stringify(payload));
    return;
  }}

  try {{
    await fetch(ARGUS_CONFIG.uploadEndpoint, {{
      method: "POST",
      headers: {{
        "Content-Type": "application/json",
        "X-Argus-API-Key": ARGUS_CONFIG.apiKey
      }},
      body: JSON.stringify(payload),
      keepalive: true
    }});
  }} catch (err) {{
    console.error("[Argus] Upload failed:", err);
  }}
}}
'''

    def generate_react_hook(self, api_key: str = "", project_id: str = "") -> str:
        """Generate React hook for recording.

        Args:
            api_key: API key
            project_id: Project ID

        Returns:
            React hook code
        """
        return f'''// Argus E2E Recording - React Hook
// Install: npm install rrweb

import {{ useEffect, useRef }} from 'react';
import {{ record }} from 'rrweb';

const ARGUS_CONFIG = {{
  apiKey: "{api_key}",
  projectId: "{project_id}",
  uploadEndpoint: "{self.config.upload_endpoint}"
}};

export function useArgusRecording(enabled = true) {{
  const eventsRef = useRef<any[]>([]);
  const sessionIdRef = useRef<string>("");

  useEffect(() => {{
    if (!enabled) return;

    sessionIdRef.current = `rec_${{Date.now()}}_${{Math.random().toString(36).substr(2, 9)}}`;

    const stopFn = record({{
      emit(event) {{
        eventsRef.current.push(event);
      }},
      maskAllInputs: {str(self.config.mask_all_inputs).lower()},
      maskTextSelector: "{self.config.mask_text_selector}"
    }});

    const interval = setInterval(() => {{
      if (eventsRef.current.length > 0) {{
        uploadEvents(eventsRef.current.splice(0), sessionIdRef.current);
      }}
    }}, {self.config.upload_interval_ms});

    return () => {{
      stopFn();
      clearInterval(interval);
      uploadEvents(eventsRef.current, sessionIdRef.current, true);
    }};
  }}, [enabled]);

  return sessionIdRef.current;
}}

async function uploadEvents(events: any[], sessionId: string, sync = false) {{
  if (events.length === 0) return;

  const payload = {{
    session_id: sessionId,
    project_id: ARGUS_CONFIG.projectId,
    events,
    metadata: {{
      href: window.location.href,
      timestamp: new Date().toISOString()
    }}
  }};

  if (sync && navigator.sendBeacon) {{
    navigator.sendBeacon(ARGUS_CONFIG.uploadEndpoint, JSON.stringify(payload));
    return;
  }}

  await fetch(ARGUS_CONFIG.uploadEndpoint, {{
    method: "POST",
    headers: {{
      "Content-Type": "application/json",
      "X-Argus-API-Key": ARGUS_CONFIG.apiKey
    }},
    body: JSON.stringify(payload)
  }});
}}
'''

    def generate_gtm_snippet(self) -> str:
        """Generate Google Tag Manager compatible snippet.

        Returns:
            GTM custom HTML tag code
        """
        return f'''<!-- Argus E2E Recording - GTM Tag -->
<script>
(function() {{
  // Configure these in GTM variables
  var ARGUS_API_KEY = {{{{Argus API Key}}}};
  var ARGUS_PROJECT_ID = {{{{Argus Project ID}}}};

  if (!ARGUS_API_KEY) {{
    console.warn("[Argus] Missing API key");
    return;
  }}

  var script = document.createElement("script");
  script.src = "{self.RRWEB_CDN}";
  script.async = true;
  script.onload = function() {{
    var events = [];
    var sessionId = "rec_" + Date.now();

    rrweb.record({{
      emit: function(event) {{ events.push(event); }},
      maskAllInputs: true
    }});

    setInterval(function() {{
      if (events.length > 0) {{
        fetch("{self.config.upload_endpoint}", {{
          method: "POST",
          headers: {{
            "Content-Type": "application/json",
            "X-Argus-API-Key": ARGUS_API_KEY
          }},
          body: JSON.stringify({{
            session_id: sessionId,
            project_id: ARGUS_PROJECT_ID,
            events: events.splice(0),
            metadata: {{ href: location.href }}
          }})
        }});
      }}
    }}, {self.config.upload_interval_ms});
  }};
  document.head.appendChild(script);
}})();
</script>
'''


def generate_snippet(
    format: str = "inline",
    api_key: str = "",
    project_id: str = "",
    config: RecorderConfig | None = None,
) -> str:
    """Generate recording snippet in specified format.

    Args:
        format: Snippet format ("inline", "npm", "react", "gtm")
        api_key: API key
        project_id: Project ID
        config: Recorder configuration

    Returns:
        Generated snippet code
    """
    generator = RecorderSnippetGenerator(config)

    if format == "inline":
        return generator.generate_inline_snippet(api_key, project_id)
    elif format == "npm":
        return generator.generate_npm_snippet(api_key, project_id)
    elif format == "react":
        return generator.generate_react_hook(api_key, project_id)
    elif format == "gtm":
        return generator.generate_gtm_snippet()
    else:
        raise ValueError(f"Unknown snippet format: {format}")
