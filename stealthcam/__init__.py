"""Stealthcam Revolver 402 human-activity detection pipeline.

Stages: poll -> YOLO11n detect -> VLM describe -> memory decision -> alert
(Mattermost) + dashboard (camera-watcher /browser).
"""

__version__ = "0.1.0"

# The decider string written into camera-watcher's Labeling.decider column.
DECIDER = "yolo11n+vlm"
