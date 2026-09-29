"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Temporal Plate Aggregator & Track Association
=============================================================================
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

from src.anpr.plate_parser import ParsedPlate


@dataclass
class VehiclePlateRecord:
    """Consolidated ANPR record for a specific vehicle track ID."""
    vehicle_track_id: int
    best_plate_text: str
    best_plate_formatted: str
    confidence: float
    is_locked: bool = False
    first_detected_frame: int = 0
    last_detected_frame: int = 0
    readings_count: int = 0
    raw_history: List[str] = field(default_factory=list)


class PlateTracker:
    """
    Maintains temporal OCR history for each tracked vehicle, executing confidence-weighted
    voting across video frames to eliminate flickering and lock the true license plate.
    """

    def __init__(
        self,
        min_votes_to_lock: int = 3,
        lock_confidence_threshold: float = 0.75,
        max_history_per_vehicle: int = 30,
    ):
        """
        :param min_votes_to_lock: Minimum consistent readings before locking plate identity.
        :param lock_confidence_threshold: Confidence score needed to finalize lock.
        :param max_history_per_vehicle: Max readings kept in memory per vehicle.
        """
        self.min_votes_to_lock = min_votes_to_lock
        self.lock_confidence_threshold = lock_confidence_threshold
        self.max_history_per_vehicle = max_history_per_vehicle

        # track_id -> dict of {plate_text: total_weighted_score}
        self._votes: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # track_id -> VehiclePlateRecord
        self.plate_records: Dict[int, VehiclePlateRecord] = {}

    def update(
        self,
        vehicle_track_id: int,
        parsed_plate: ParsedPlate,
        frame_id: int,
    ) -> Optional[VehiclePlateRecord]:
        """
        Update vehicle record with a newly parsed plate reading.

        :param vehicle_track_id: Persistent ID from ByteTrack.
        :param parsed_plate: ParsedPlate result from current frame.
        :param frame_id: Current frame sequence index.
        :return: Current VehiclePlateRecord for this vehicle.
        """
        if not parsed_plate.cleaned_text or len(parsed_plate.cleaned_text) < 4:
            return self.plate_records.get(vehicle_track_id)

        # Weight score by validity and OCR confidence
        weight = parsed_plate.confidence_score * (1.5 if parsed_plate.is_valid else 0.8)
        self._votes[vehicle_track_id][parsed_plate.cleaned_text] += weight

        # Initialize or retrieve record
        if vehicle_track_id not in self.plate_records:
            self.plate_records[vehicle_track_id] = VehiclePlateRecord(
                vehicle_track_id=vehicle_track_id,
                best_plate_text=parsed_plate.cleaned_text,
                best_plate_formatted=parsed_plate.formatted_text,
                confidence=parsed_plate.confidence_score,
                is_locked=False,
                first_detected_frame=frame_id,
                last_detected_frame=frame_id,
                readings_count=1,
                raw_history=[parsed_plate.cleaned_text],
            )
        else:
            rec = self.plate_records[vehicle_track_id]
            rec.last_detected_frame = frame_id
            rec.readings_count += 1
            rec.raw_history.append(parsed_plate.cleaned_text)
            if len(rec.raw_history) > self.max_history_per_vehicle:
                rec.raw_history.pop(0)

        record = self.plate_records[vehicle_track_id]

        # If already locked with high confidence, maintain lock unless overwhelmingly contradicted
        if record.is_locked and record.confidence > 0.90:
            return record

        # Determine highest scoring candidate plate text
        vote_dict = self._votes[vehicle_track_id]
        best_candidate, best_score = max(vote_dict.items(), key=lambda item: item[1])

        # Normalize score into confidence (0.0 to 1.0)
        norm_conf = min(1.0, best_score / max(1, record.readings_count))

        record.best_plate_text = best_candidate
        # Retain formatted version if matches candidate
        if parsed_plate.cleaned_text == best_candidate:
            record.best_plate_formatted = parsed_plate.formatted_text
        record.confidence = round(norm_conf, 4)

        # Check lock conditions
        if (
            record.readings_count >= self.min_votes_to_lock
            and record.confidence >= self.lock_confidence_threshold
        ):
            record.is_locked = True

        return record

    def get_plate(self, vehicle_track_id: int) -> Optional[VehiclePlateRecord]:
        """Retrieve the best current plate record for a given vehicle."""
        return self.plate_records.get(vehicle_track_id)

    def cleanup_old_tracks(self, active_track_ids: List[int]) -> None:
        """Purge records for tracks no longer active."""
        active_set = set(active_track_ids)
        to_remove = [t_id for t_id in self.plate_records if t_id not in active_set]
        for t_id in to_remove:
            del self.plate_records[t_id]
            del self._votes[t_id]
