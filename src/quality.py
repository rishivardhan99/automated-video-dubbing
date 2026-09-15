import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum


class QualityState(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class QualityReport:
    state: QualityState
    diagnostics: dict
    warnings: list[str]
    failures: list[str]

    def log(self, logger):
        """Log the report to a given logger."""
        logger.info("=== Transcript Quality Report ===")
        logger.info(f"State: {self.state.value}")
        for k, v in self.diagnostics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.2f}")
            else:
                logger.info(f"  {k}: {v}")
        
        if self.warnings:
            logger.warning("Warnings:")
            for w in self.warnings:
                logger.warning(f"  - {w}")
                
        if self.failures:
            logger.error("Failures:")
            for f in self.failures:
                logger.error(f"  - {f}")
        logger.info("=================================")


def analyze_transcript(transcript: dict, audio_duration: float, processing_time: float) -> QualityReport:
    """
    Analyze transcript quality and return PASS/WARN/FAIL with diagnostics.
    """
    segments = transcript.get("segments", [])
    text = transcript.get("text", "")
    language = transcript.get("language")
    confidence = transcript.get("language_probability")
    
    warnings = []
    failures = []
    
    # Base diagnostics
    segment_count = len(segments)
    rtf = processing_time / audio_duration if audio_duration > 0 else 0.0
    
    if segment_count == 0:
        failures.append("Transcript contains no segments.")
        return QualityReport(
            state=QualityState.FAIL,
            diagnostics={"duration": audio_duration, "processing_time": processing_time, "rtf": rtf},
            warnings=warnings,
            failures=failures,
        )

    # Derived metrics
    segment_texts = [s.get("text", "").strip() for s in segments]
    avg_len = sum(len(t) for t in segment_texts) / segment_count
    avg_dur = sum(s["end"] - s["start"] for s in segments) / segment_count
    
    single_char_count = sum(1 for t in segment_texts if len(t) <= 2)
    single_char_pct = (single_char_count / segment_count) * 100
    
    # Near-zero duration segments
    zero_dur_count = sum(1 for s in segments if s["end"] - s["start"] < 0.1)
    
    # Repetition
    counts = Counter(segment_texts)
    most_repeated_count = counts.most_common(1)[0][1] if counts else 0
    repetition_pct = (most_repeated_count / segment_count) * 100
    
    # Script detection
    script_counts = {}
    for t in segment_texts:
        for ch in t:
            if ch.isalpha():
                try:
                    script = unicodedata.name(ch, "UNKNOWN").split()[0]
                    script_counts[script] = script_counts.get(script, 0) + 1
                except ValueError:
                    pass

    total_chars = sum(script_counts.values())
    script_families = {s: c for s, c in script_counts.items()}
    num_scripts = len(script_families)
    
    # --- Check Rules ---
    
    # 1. Confidence
    if confidence is not None and confidence < 0.5:
        warnings.append(f"Low language confidence ({confidence:.2f})")
        
    # 2. Extreme repetition
    is_highly_repeated = False
    if repetition_pct > 20.0 and segment_count > 10:
        is_highly_repeated = True
        warnings.append(f"High repetition detected: a segment repeats {repetition_pct:.1f}% of the time.")
    if repetition_pct > 40.0 and segment_count > 10:
        failures.append(f"Pathological repetition: a segment repeats {repetition_pct:.1f}% of the time.")
        
    # 3. Tiny segments
    has_many_tiny_segments = False
    if single_char_pct > 15.0:
        has_many_tiny_segments = True
        warnings.append(f"High number of very short segments ({single_char_pct:.1f}%).")
    if single_char_pct > 30.0:
        failures.append(f"Excessive number of very short segments ({single_char_pct:.1f}%).")
        
    if zero_dur_count > segment_count * 0.1 and segment_count > 10:
        warnings.append(f"Many near-zero duration segments ({zero_dur_count}).")

    # 4. Timestamp validity
    invalid_ts = sum(
        1 for s in segments
        if s.get("end", 0) < s.get("start", 0)
    )
    if invalid_ts > 0:
        failures.append(
            f"Invalid timestamps: {invalid_ts} "
            f"segments have end < start."
        )

    # 5. Excessive segment density
    if audio_duration > 0 and (segment_count / audio_duration) > 1.5:
        failures.append(f"Excessive segment count relative to duration ({segment_count} segs in {audio_duration:.1f}s).")

    # 6. Intra-segment repetition (hallucination loops)
    intra_rep_count = 0
    for t in segment_texts:
        if len(t) >= 10:
            # Check if a short substring repeats excessively
            for plen in range(1, 4):
                if len(t) >= plen * 5:
                    pattern = t[:plen]
                    if t == pattern * (len(t) // plen) + pattern[:len(t) % plen]:
                        intra_rep_count += 1
                        break
    if intra_rep_count > 0:
        pct = (intra_rep_count / segment_count) * 100
        if pct > 30:
            failures.append(
                f"Intra-segment repetition loops "
                f"detected in {intra_rep_count} "
                f"segments ({pct:.0f}%)."
            )
        elif pct > 10:
            warnings.append(
                f"Intra-segment repetition in "
                f"{intra_rep_count} segments "
                f"({pct:.0f}%)."
            )
        
    # 5. Script mixing
    if num_scripts > 2:
        # Check if the secondary scripts make up a significant portion of the text
        sorted_scripts = sorted(script_families.items(), key=lambda x: -x[1])
        main_script, main_count = sorted_scripts[0]
        
        # If there's garbage script mixing
        if (is_highly_repeated or has_many_tiny_segments) and num_scripts >= 4:
            failures.append(f"Severe script mixing combined with structural corruption. Scripts: {list(script_families.keys())}")
        else:
            warnings.append(f"Multiple scripts detected: {list(script_families.keys())}")
            
    diagnostics = {
        "audio_duration_s": audio_duration,
        "processing_time_s": processing_time,
        "real_time_factor": rtf,
        "language": language,
        "confidence": confidence,
        "segment_count": segment_count,
        "avg_segment_duration_s": avg_dur,
        "avg_segment_chars": avg_len,
        "single_char_pct": single_char_pct,
        "repetition_pct": repetition_pct,
        "script_families": script_families,
    }

    if failures:
        state = QualityState.FAIL
    elif warnings:
        state = QualityState.WARN
    else:
        state = QualityState.PASS
        
    return QualityReport(state=state, diagnostics=diagnostics, warnings=warnings, failures=failures)
