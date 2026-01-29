"""Analytics integration plugins.

This module contains plugins for analytics platforms:
- Amplitude: Product analytics for user behavior tracking
- Mixpanel: User analytics with funnels and cohorts
- Segment: Customer data platform for data routing

Each plugin extends IntegrationPlugin and provides:
- test_connection(): Validate API credentials
- sync_data(): Fetch events, cohorts, and configuration
- Platform-specific features (event tracking, exports, etc.)
"""

from .amplitude_plugin import AmplitudePlugin
from .mixpanel_plugin import MixpanelPlugin
from .segment_plugin import SegmentPlugin

__all__ = [
    "AmplitudePlugin",
    "MixpanelPlugin",
    "SegmentPlugin",
]
