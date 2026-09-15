"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Object Identity & State Management
=============================================================================
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import time


@dataclass
class TrackedObject:
    """
    Data contract representing a uniquely tracked surveillance entity across frames.
    """
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]       # [x1, y1, x2, y2] in pixel coordinates
    center: List[float]     # [center_x, center_y]
    frame_id: int           # Frame sequence index
    timestamp: float = field(default_factory=time.time)

    @property
    def bottom_center(self) -> List[float]:
        """
        Returns [x, y2], the ground-contact point of the bounding box.
        Crucial for virtual fence intrusion, tripwire collision, and ground plane tracking.
        """
        return [self.center[0], self.bbox[3]]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> Dict:
        """Convert to standard serializable dictionary."""
        return asdict(self)


class ObjectIdentityManager:
    """
    Maintains lifecycle and persistent metadata for all active and historical tracks.
    """

    def __init__(self, max_lost_frames: int = 30):
        """
        :param max_lost_frames: Number of consecutive missing frames before considering a track expired.
        """
        self.max_lost_frames = max_lost_frames
        self.active_tracks: Dict[int, TrackedObject] = {}
        self.track_metadata: Dict[int, Dict] = {}

    def update(self, tracked_objects: List[TrackedObject], frame_id: int) -> Dict[int, TrackedObject]:
        """
        Update registry with newly detected/tracked objects in the current frame.
        """
        current_active_ids = set()

        for obj in tracked_objects:
            t_id = obj.track_id
            current_active_ids.add(t_id)
            self.active_tracks[t_id] = obj

            if t_id not in self.track_metadata:
                self.track_metadata[t_id] = {
                    "first_frame": frame_id,
                    "first_seen_time": obj.timestamp,
                    "last_frame": frame_id,
                    "last_seen_time": obj.timestamp,
                    "class_name": obj.class_name,
                    "class_id": obj.class_id,
                    "total_detections": 1,
                }
            else:
                meta = self.track_metadata[t_id]
                meta["last_frame"] = frame_id
                meta["last_seen_time"] = obj.timestamp
                meta["total_detections"] += 1

        # Purge stale tracks that exceeded max_lost_frames
        expired_ids = [
            t_id for t_id, obj in self.active_tracks.items()
            if (frame_id - obj.frame_id) > self.max_lost_frames
        ]
        for t_id in expired_ids:
            del self.active_tracks[t_id]

        return self.active_tracks

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        return self.active_tracks.get(track_id)

    def get_metadata(self, track_id: int) -> Optional[Dict]:
        return self.track_metadata.get(track_id)
