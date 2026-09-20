"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: License Plate Parser, Character Normalizer & Regex Validator
=============================================================================
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ParsedPlate:
    """Structured representation of a validated license plate."""
    raw_text: str
    cleaned_text: str
    formatted_text: str
    is_valid: bool
    state_code: Optional[str] = None
    series: Optional[str] = None
    registration_num: Optional[str] = None
    confidence_score: float = 0.0


class PlateParser:
    """
    Cleans raw OCR output, resolves common optical character confusions,
    and validates against vehicle registration patterns (Standard & BH Series).
    """

    # Common OCR character confusions: Alphabet <-> Digit
    CHAR_TO_DIGIT = {
        "O": "0", "D": "0", "Q": "0",
        "I": "1", "L": "1", "T": "1", "|": "1",
        "Z": "2",
        "E": "3",
        "A": "4",
        "S": "5",
        "G": "6", "b": "6",
        "B": "8",
        "g": "9", "q": "9",
    }

    DIGIT_TO_CHAR = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "3": "E",
        "4": "A",
        "5": "S",
        "6": "G",
        "8": "B",
    }

    # Indian State & Union Territory Codes (relevant for border & highway surveillance)
    INDIAN_STATE_CODES = {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DN", "DL", "GA", "GJ",
        "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML",
        "MZ", "NL", "OD", "PY", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"
    }

    # Regex patterns
    # Standard: e.g., DL01AB1234 or HR26DK8337
    REGEX_STANDARD = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$")
    # BH Series: e.g., 22BH1234AA
    REGEX_BH = re.compile(r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$")
    # Generic Alphanumeric Fallback (5 to 11 characters)
    REGEX_GENERIC = re.compile(r"^[A-Z0-9]{5,12}$")

    def __init__(self):
        pass

    def clean_text(self, raw_text: str) -> str:
        """
        Strip whitespace, punctuation, and non-alphanumeric noise from OCR string.
        """
        if not raw_text:
            return ""
        # Remove anything that is not A-Z, a-z, 0-9
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        return cleaned

    def parse(self, raw_text: str, ocr_confidence: float = 0.0) -> ParsedPlate:
        """
        Parse and validate raw OCR string into a structured ParsedPlate.

        :param raw_text: Raw string emitted by OCR engine.
        :param ocr_confidence: Confidence score from OCR engine (0.0 to 1.0).
        :return: ParsedPlate instance.
        """
        cleaned = self.clean_text(raw_text)
        if len(cleaned) < 4:
            return ParsedPlate(
                raw_text=raw_text,
                cleaned_text=cleaned,
                formatted_text=cleaned,
                is_valid=False,
                confidence_score=ocr_confidence * 0.2,
            )

        # Attempt Standard Pattern matching
        match_std = self.REGEX_STANDARD.match(cleaned)
        if match_std:
            state, district, series, number = match_std.groups()
            is_valid_state = state in self.INDIAN_STATE_CODES
            formatted = f"{state} {district} {series} {number}".replace("  ", " ").strip()
            score = ocr_confidence * (1.0 if is_valid_state else 0.8)
            return ParsedPlate(
                raw_text=raw_text,
                cleaned_text=cleaned,
                formatted_text=formatted,
                is_valid=True,
                state_code=state,
                series=series if series else None,
                registration_num=number,
                confidence_score=round(score, 4),
            )

        # Attempt BH Series matching
        match_bh = self.REGEX_BH.match(cleaned)
        if match_bh:
            year, bh, number, series = match_bh.groups()
            formatted = f"{year} {bh} {number} {series}".strip()
            return ParsedPlate(
                raw_text=raw_text,
                cleaned_text=cleaned,
                formatted_text=formatted,
                is_valid=True,
                state_code="BH",
                series=series,
                registration_num=number,
                confidence_score=round(ocr_confidence * 0.95, 4),
            )

        # Apply Heuristic Confusion Correction if close to standard length (9-10 chars)
        corrected = self._correct_standard_plate(cleaned)
        match_corr = self.REGEX_STANDARD.match(corrected)
        if match_corr:
            state, district, series, number = match_corr.groups()
            is_valid_state = state in self.INDIAN_STATE_CODES
            formatted = f"{state} {district} {series} {number}".replace("  ", " ").strip()
            return ParsedPlate(
                raw_text=raw_text,
                cleaned_text=corrected,
                formatted_text=formatted,
                is_valid=True,
                state_code=state,
                series=series if series else None,
                registration_num=number,
                confidence_score=round(ocr_confidence * (0.90 if is_valid_state else 0.70), 4),
            )

        # Fallback: Generic alphanumeric validity
        is_generic_valid = bool(self.REGEX_GENERIC.match(cleaned))
        return ParsedPlate(
            raw_text=raw_text,
            cleaned_text=cleaned,
            formatted_text=cleaned,
            is_valid=is_generic_valid,
            confidence_score=round(ocr_confidence * 0.50, 4),
        )

    def _correct_standard_plate(self, text: str) -> str:
        """
        Heuristically corrects characters at known positional slots (State, Number, Series).
        """
        if len(text) not in [9, 10]:
            return text

        chars = list(text)

        # State code positions (chars 0, 1) -> must be alphabets
        for i in [0, 1]:
            if chars[i].isdigit() and chars[i] in self.DIGIT_TO_CHAR:
                chars[i] = self.DIGIT_TO_CHAR[chars[i]]

        # District code positions (chars 2, 3) -> must be digits
        for i in [2, 3]:
            if chars[i].isalpha() and chars[i] in self.CHAR_TO_DIGIT:
                chars[i] = self.CHAR_TO_DIGIT[chars[i]]

        # Last 4 positions -> must be digits
        for i in range(len(chars) - 4, len(chars)):
            if chars[i].isalpha() and chars[i] in self.CHAR_TO_DIGIT:
                chars[i] = self.CHAR_TO_DIGIT[chars[i]]

        return "".join(chars)


if __name__ == "__main__":
    parser = PlateParser()
    sample_tests = ["DL01AB1234", "dl-1-ab-1234", "HR26DK8337", "22BH9999AA", "IND DL 01 AB 1234", "0L01AB1234"]
    print("[IBVAP-ANPR] Running PlateParser tests:")
    for sample in sample_tests:
        res = parser.parse(sample, ocr_confidence=0.92)
        print(f"  • Raw: '{sample}' -> Formatted: '{res.formatted_text}' (Valid: {res.is_valid}, Conf: {res.confidence_score})")
