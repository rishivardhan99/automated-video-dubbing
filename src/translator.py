import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.utils.logger import get_logger
from src.utils.paths import TRANSLATION_BATCHES_DIR, TRANSLATIONS_DIR

load_dotenv()

logger = get_logger(__name__)


class TranslationError(Exception):
    """Raised when translation fails."""


class GroqTranslator:
    def __init__(
        self,
        model_name: str | None = None,
        max_retries: int = 5,
    ) -> None:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise TranslationError(
                "GROQ_API_KEY is not configured. "
                "Add it to your .env file."
            )

        self.model_name = (
            model_name
            or os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            )
        )
        
        try:
            self.max_segments = int(os.getenv("TRANSLATION_MAX_SEGMENTS", "12"))
            self.max_chars = int(os.getenv("TRANSLATION_MAX_CHARS", "3000"))
        except ValueError:
            self.max_segments = 12
            self.max_chars = 3000

        self.max_retries = max_retries

        self.client = Groq(api_key=api_key)

    def translate(
        self,
        transcript_path: Path,
    ) -> Path:

        if not transcript_path.exists():
            raise TranslationError(
                f"Transcript does not exist: {transcript_path}"
            )

        with transcript_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            transcript = json.load(file)

        source_language = transcript.get("language")
        segments = transcript.get("segments", [])

        if not segments:
            raise TranslationError(
                "Transcript contains no segments."
            )

        logger.info(
            "Source language: %s",
            source_language,
        )

        logger.info(
            "Segments to translate: %d",
            len(segments),
        )

        # English requires no translation.
        if source_language == "en":
            logger.info(
                "Source is already English. "
                "Skipping Groq translation."
            )

            translated_segments = []
            for segment in segments:
                seg = {
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "source_text": segment["text"],
                    "translated_text": segment["text"],
                }
                # Preserve speaker metadata from diarization
                if "speaker" in segment:
                    seg["speaker"] = segment["speaker"]
                if "speaker_confidence" in segment:
                    seg["speaker_confidence"] = segment["speaker_confidence"]
                translated_segments.append(seg)

        else:
            video_id = transcript_path.stem
            translated_segments = (
                self._translate_segments(
                    segments,
                    source_language,
                    video_id
                )
            )

        result = {
            "model": self.model_name,
            "source_language": source_language,
            "target_language": "en",
            "segments": translated_segments,
        }

        TRANSLATIONS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            TRANSLATIONS_DIR
            / f"{transcript_path.stem}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("Translation complete.")

        logger.info(
            "Saved translation: %s",
            output_path,
        )

        return output_path

    def _build_batches(
        self,
        segments: list[dict],
    ) -> list[list[dict]]:

        batches = []
        current_batch = []
        current_chars = 0

        for segment in segments:
            text = segment.get("text", "")
            text_length = len(text)
            
            if not current_batch:
                current_batch.append(segment)
                current_chars += text_length
            else:
                if len(current_batch) >= self.max_segments or (current_chars + text_length > self.max_chars):
                    batches.append(current_batch)
                    current_batch = [segment]
                    current_chars = text_length
                else:
                    current_batch.append(segment)
                    current_chars += text_length

        if current_batch:
            batches.append(current_batch)

        return batches
        
    def _translate_batch_adaptive(
        self,
        batch_segments: list[dict],
        source_language: str | None,
        context_before: list[dict],
        context_after: list[dict],
        batch_cache_path: Path | None,
    ) -> list[dict]:
        """
        Translates a batch, bisecting it into smaller batches if it fails repeatedly.
        """
        # Check cache
        if batch_cache_path and batch_cache_path.exists():
            try:
                with batch_cache_path.open("r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if isinstance(cached_data, list):
                    logger.info("Loaded translation batch from cache: %s", batch_cache_path.name)
                    return cached_data
            except Exception as e:
                logger.warning("Failed to load translation batch cache %s: %s", batch_cache_path, e)

        try:
            result = self._translate_batch(batch_segments, source_language, context_before, context_after)
            # Save cache
            if batch_cache_path:
                batch_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with batch_cache_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            return result
        except TranslationError as exc:
            if len(batch_segments) > 1:
                mid = len(batch_segments) // 2
                logger.warning(
                    "Batch failed, bisecting adaptive batch: %d -> %d + %d. Reason: %s", 
                    len(batch_segments), mid, len(batch_segments) - mid, exc
                )
                first_half = batch_segments[:mid]
                second_half = batch_segments[mid:]
                
                # Context for first half includes the start of second half
                res1 = self._translate_batch_adaptive(
                    first_half, 
                    source_language, 
                    context_before, 
                    second_half[:2], 
                    None # don't cache intermediate bisections to keep it simple, or cache to main
                )
                
                # Context for second half includes the end of first half
                res2 = self._translate_batch_adaptive(
                    second_half, 
                    source_language, 
                    first_half[-2:], 
                    context_after, 
                    None
                )
                
                combined = res1 + res2
                
                # Cache the combined success
                if batch_cache_path:
                    with batch_cache_path.open("w", encoding="utf-8") as f:
                        json.dump(combined, f, ensure_ascii=False, indent=2)
                
                return combined
            else:
                logger.error("Cannot bisect single segment! ID: %s", batch_segments[0].get("id"))
                raise

    def _translate_segments(
        self,
        segments: list[dict],
        source_language: str | None,
        video_id: str,
    ) -> list[dict]:

        batches = self._build_batches(segments)

        logger.info(
            "Translation batches: %d",
            len(batches),
        )

        translated_segments = []
        batch_cache_dir = TRANSLATION_BATCHES_DIR / video_id
        batch_cache_dir.mkdir(parents=True, exist_ok=True)

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):
            logger.info(
                "Translating batch %d/%d (%d segments)...",
                batch_number,
                len(batches),
                len(batch),
            )
            
            # Find context windows
            idx_start = segments.index(batch[0])
            idx_end = segments.index(batch[-1])
            
            context_before = segments[max(0, idx_start - 2):idx_start]
            context_after = segments[idx_end + 1:idx_end + 3]
            
            batch_cache_path = batch_cache_dir / f"batch_{batch_number:03d}.json"

            translated_batch = self._translate_batch_adaptive(
                batch_segments=batch,
                source_language=source_language,
                context_before=context_before,
                context_after=context_after,
                batch_cache_path=batch_cache_path
            )

            translated_segments.extend(
                translated_batch
            )

        # Final global validation.
        expected_ids = [
            segment["id"]
            for segment in segments
        ]

        actual_ids = [
            segment["id"]
            for segment in translated_segments
        ]

        if expected_ids != actual_ids:
            raise TranslationError(
                "Final translation changed segment order "
                "or removed/duplicated segments."
            )

        return translated_segments

    def _translate_batch(
        self,
        segments: list[dict],
        source_language: str | None,
        context_before: list[dict],
        context_after: list[dict],
    ) -> list[dict]:

        target_payload = [
            {
                "id": segment["id"],
                "text": segment["text"],
            }
            for segment in segments
        ]
        
        ctx_before_payload = [{"id": s["id"], "text": s["text"]} for s in context_before]
        ctx_after_payload = [{"id": s["id"], "text": s["text"]} for s in context_after]

        prompt = f"""
Translate spoken dialogue from
{source_language or "the detected source language"}
into natural spoken English.

This translation will be used for automated video dubbing.

The source transcript may contain phonetic spellings or minor ASR errors.
Use surrounding segments and contextual clues to reconstruct obvious
speech-recognition errors.

Examples:
- phonetic renderings of English words may need to be restored to English
- names, cities, exams, organizations, and technical terms should be
  recognized when context strongly supports them

Rules:
- do not invent facts
- do not add information not reasonably supported by the source/context
- preserve the speaker's intent
- prefer natural spoken English over literal translation
- keep the result concise enough for the original timestamp
- preserve every segment ID exactly
- return exactly one translation for every target segment
- never merge segments
- never split segments
- preserve the original segment order
- CRITICAL: Never return meta-commentary like "I don't understand" 
  or "Inaudible". If a segment is complete gibberish, infer from context, 
  or return an empty string "".

### CONTEXT BEFORE (DO NOT TRANSLATE THESE):
{json.dumps(ctx_before_payload, ensure_ascii=False, indent=2) if ctx_before_payload else "None"}

### TARGET SEGMENTS (TRANSLATE ONLY THESE):
{json.dumps(target_payload, ensure_ascii=False, indent=2)}

### CONTEXT AFTER (DO NOT TRANSLATE THESE):
{json.dumps(ctx_after_payload, ensure_ascii=False, indent=2) if ctx_after_payload else "None"}

Return translations ONLY for the TARGET SEGMENTS using the required JSON schema.
"""

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            try:

                response = (
                    self.client.chat.completions.create(
                        model=self.model_name,

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a professional "
                                    "multilingual video-dubbing translator. "
                                    "Return only the requested "
                                    "structured data."
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],

                        temperature=0.1,

                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "translation_response",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "translations": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {
                                                        "type": "integer"
                                                    },
                                                    "translated_text": {
                                                        "type": "string"
                                                    },
                                                },
                                                "required": [
                                                    "id",
                                                    "translated_text",
                                                ],
                                                "additionalProperties": False,
                                            },
                                        }
                                    },
                                    "required": [
                                        "translations"
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                        },

                        reasoning_effort="low",
                    )
                )

                content = (
                    response.choices[0]
                    .message
                    .content
                )

                if not content:
                    raise TranslationError(
                        "Groq returned empty content."
                    )

                parsed = json.loads(content)

                translated = parsed.get(
                    "translations"
                )

                if not isinstance(
                    translated,
                    list,
                ):
                    raise TranslationError(
                        "Groq returned an invalid "
                        "translations array."
                    )

                return self._validate_batch(
                    segments,
                    translated,
                )

            except Exception as exc:

                logger.warning(
                    "Translation attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

                if attempt >= self.max_retries:
                    raise TranslationError(
                        "Groq translation failed after "
                        f"{self.max_retries} attempts: {exc}"
                    ) from exc

                wait_seconds = self._get_retry_delay(
                    exc,
                    attempt,
                )

                logger.info(
                    "Waiting %.2f seconds before retry...",
                    wait_seconds,
                )

                time.sleep(wait_seconds)

        raise TranslationError(
            "Translation failed unexpectedly."
        )

    @staticmethod
    def _get_retry_delay(
        exc: Exception,
        attempt: int,
    ) -> float:
        import re

        # Conservative exponential backoff.
        base_delay = min(
            2 ** (attempt - 1),
            30,
        )

        message = str(exc).lower()

        # Check for Groq's "please try again in XmYs" or "please try again in Xs" format
        match = re.search(r"please try again in (?:(\d+)m)?([\d\.]+)s", message)
        if match:
            minutes = int(match.group(1)) if match.group(1) else 0
            seconds = float(match.group(2))
            total_seconds = (minutes * 60) + seconds
            return max(total_seconds + 1.0, float(base_delay))

        # Groq's 429 response may expose
        # a retry-after duration.
        for token in [
            "retry-after:",
            "retry after",
            "x-ratelimit-reset"
        ]:

            if token in message:
                try:
                    after = (
                        message
                        .split(token, 1)[1]
                        .split()[0]
                    )
                    after = ''.join(c for c in after if c.isdigit() or c == '.')
                    return max(
                        float(after),
                        float(base_delay),
                    )

                except (
                    ValueError,
                    IndexError,
                ):
                    pass

        import random
        return float(base_delay) + random.uniform(0.1, 1.0)

    @staticmethod
    def _validate_batch(
        source_segments: list[dict],
        translated_segments: list[dict],
    ) -> list[dict]:

        expected_ids = [
            segment["id"]
            for segment in source_segments
        ]

        received_ids = [
            segment.get("id")
            for segment in translated_segments
        ]

        if expected_ids != received_ids:
            raise TranslationError(
                "Groq changed, removed, duplicated, "
                "or reordered segment IDs."
            )

        if len(translated_segments) != len(
            source_segments
        ):
            raise TranslationError(
                "Groq returned the wrong number "
                "of translations."
            )

        result = []

        for source, translation in zip(
            source_segments,
            translated_segments,
        ):

            translated_text = translation.get(
                "translated_text"
            )

            if not isinstance(
                translated_text,
                str,
            ):
                raise TranslationError(
                    f"Invalid translation for "
                    f"segment {source['id']}."
                )

            translated_text = (
                translated_text.strip()
            )

            if translated_text == "":
                # We allow empty strings for complete gibberish now.
                pass
            elif not translated_text:
                raise TranslationError(
                    f"Empty or missing translation for "
                    f"segment {source['id']}."
                )

            result.append(
                {
                    "id": source["id"],
                    "start": source["start"],
                    "end": source["end"],
                    "source_text": source["text"],
                    "translated_text": translated_text,
                    # Preserve speaker metadata from diarization
                    **({
                        "speaker": source["speaker"],
                    } if "speaker" in source else {}),
                    **({
                        "speaker_confidence": source["speaker_confidence"],
                    } if "speaker_confidence" in source else {}),
                }
            )

        return result