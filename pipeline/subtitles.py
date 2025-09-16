"""Subtitle and metadata utilities extracted from :mod:`video_processor`."""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from temp_function import _llm_generate_caption_hashtags_fixed


def _format_srt_timestamp(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(segments: Sequence[Dict], srt_path: Path) -> None:
    srt_path = Path(srt_path)
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    index = 1
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or max(0.0, start + 0.01))
        start_ts = _format_srt_timestamp(start)
        end_ts = _format_srt_timestamp(end)
        lines.append(str(index))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
        index += 1
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_vtt(segments: Sequence[Dict], vtt_path: Path) -> None:
    vtt_path = Path(vtt_path)
    vtt_path.parent.mkdir(parents=True, exist_ok=True)

    def to_vtt_ts(total_seconds: float) -> str:
        if total_seconds < 0:
            total_seconds = 0.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or max(0.0, start + 0.01))
        lines.append(f"{to_vtt_ts(start)} --> {to_vtt_ts(end)}")
        lines.append(text)
        lines.append("")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class SubtitleProcessor:
    """Gestion de la transcription et des métadonnées."""

    def __init__(
        self,
        config,
        whisper_model,
        logger,
        print_func=print,
    ) -> None:
        self.config = config
        self._whisper = whisper_model
        self._logger = logger
        self._print = print_func

    def transcribe_audio(self, video_path: Path) -> str:
        self._logger.info("📝 Transcription audio avec Whisper")
        self._print("    📝 Transcription Whisper en cours...")
        result = self._whisper.transcribe(str(video_path))
        self._print("    ✅ Transcription terminée")
        return result["text"]

    def transcribe_segments(self, video_path: Path) -> List[Dict]:
        self._logger.info("⏱️ Transcription avec timestamps")
        self._print("    ⏱️ Génération des timestamps...")
        result = self._whisper.transcribe(str(video_path), word_timestamps=True)
        bias = getattr(self.config, "SUBTITLE_TIMING_BIAS_S", 0.0)
        subtitles: List[Dict] = []
        for segment in result.get("segments", []):
            seg_start = max(0.0, (segment.get("start") or 0.0) + bias)
            seg_end = max(seg_start, (segment.get("end") or seg_start) + bias)
            subtitle: Dict = {
                "text": (segment.get("text") or "").strip(),
                "start": seg_start,
                "end": seg_end,
            }
            words = segment.get("words")
            if words:
                precise_words = []
                for word in words:
                    ws = max(0.0, (word.get("start") or seg_start) + bias)
                    we = max(ws, (word.get("end") or ws) + bias)
                    wt = (word.get("word") or word.get("text") or "").strip()
                    if wt:
                        precise_words.append({"text": wt, "start": ws, "end": we})
                if precise_words:
                    subtitle["words"] = precise_words
            subtitles.append(subtitle)
        self._print(f"    ✅ {len(subtitles)} segments de sous-titres générés")
        return subtitles

    def generate_caption_and_hashtags(
        self, subtitles: List[Dict]
    ) -> Tuple[str, str, List[str], List[str]]:
        full_text = " ".join(s.get("text", "") for s in subtitles)
        try:
            sys.path.insert(0, str(Path(__file__).parent / "utils"))
            from pipeline_integration import create_pipeline_integration  # type: ignore

            llm_integration = create_pipeline_integration()
            self._print(
                f"    🚀 [LLM INDUSTRIEL] Génération de métadonnées pour {len(full_text)} caractères"
            )
            safe_timestamps = []
            for segment in subtitles:
                if isinstance(segment, dict) and "start" in segment and "end" in segment:
                    try:
                        start_val = segment.get("start", 0)
                        end_val = segment.get("end", 0)
                        if hasattr(start_val, "start"):
                            start_val = start_val.start or 0
                        if hasattr(end_val, "start"):
                            end_val = end_val.start or 0
                        safe_timestamps.append((float(start_val), float(end_val)))
                    except (ValueError, TypeError, AttributeError):
                        continue
            result = llm_integration.process_video_transcript(
                transcript=full_text,
                video_id=f"video_{int(time.time())}",
                segment_timestamps=safe_timestamps,
            )
            if result.get("success", False):
                metadata = result.get("metadata", {})
                broll_data = result.get("broll_data", {})
                title = metadata.get("title", "").strip()
                description = metadata.get("description", "").strip()
                hashtags = [h for h in (metadata.get("hashtags") or []) if h]
                broll_keywords = broll_data.get("keywords", [])
                self._print("    ✅ [LLM INDUSTRIEL] Métadonnées générées avec succès")
                self._print(f"    🎯 Titre: {title}")
                self._print(f"    📝 Description: {description[:100]}...")
                self._print(f"    #️⃣ Hashtags: {len(hashtags)} générés")
                self._print(f"    🎬 Mots-clés B-roll: {len(broll_keywords)} termes optimisés")
                return title, description, hashtags, broll_keywords
            self._print("    ⚠️ [LLM INDUSTRIEL] Échec, fallback vers ancien système")
            raise RuntimeError("LLM industriel échoué")
        except Exception as exc:
            self._print(f"    🔄 [FALLBACK] Retour vers ancien système: {exc}")
            llm_res = _llm_generate_caption_hashtags_fixed(full_text)
            if llm_res and (
                llm_res.get("title")
                or llm_res.get("description")
                or llm_res.get("hashtags")
            ):
                title = (llm_res.get("title") or "").strip()
                description = (llm_res.get("description") or "").strip()
                hashtags = [h for h in (llm_res.get("hashtags") or []) if h]
                broll_keywords = llm_res.get("broll_keywords", [])
                if broll_keywords:
                    self._print(
                        f"    🤖 [LLM] Titre/description/hashtags + {len(broll_keywords)} mots-clés B-roll générés par LLM local"
                    )
                    self._print(
                        f"    🎯 Mots-clés B-roll LLM: {', '.join(broll_keywords[:8])}..."
                    )
                else:
                    self._print(
                        "    🤖 [LLM] Titre/description/hashtags générés par LLM local"
                    )
                    fallback_text = f"{title} {description}".lower()
                    broll_keywords = [
                        word
                        for word in fallback_text.split()
                        if len(word) > 3 and word.isalpha()
                    ]
                    broll_keywords = list(set(broll_keywords))[:10]
                    self._print(
                        f"    🔄 Fallback mots-clés B-roll: {', '.join(broll_keywords[:5])}..."
                    )
                if not title and description:
                    title = description[:60] + ("…" if len(description) > 60 else "")
                return title, description, hashtags, broll_keywords
        words = [
            w.strip().lower()
            for w in re.split(r"[^a-zA-Z0-9éèàùçêîôâ]+", full_text)
            if len(w) > 2
        ]
        counts = Counter(words)
        common = [w for w, _ in counts.most_common(12) if w.isalpha()]
        hashtags = [f"#{w}" for w in common[:12]]
        text_lower = full_text.lower()
        visual_keywords: List[str] = []
        if any(
            word in text_lower
            for word in ["scientific", "research", "study", "paper", "published"]
        ):
            visual_keywords.extend(
                ["scientist", "research", "laboratory", "study", "analysis", "discovery"]
            )
        if any(
            word in text_lower
            for word in ["behavior", "cognitive", "psychology", "mental"]
        ):
            visual_keywords.extend(
                ["therapy", "counseling", "brain", "mind", "psychology", "behavior"]
            )
        if any(
            word in text_lower
            for word in ["work", "effort", "exert", "challenge"]
        ):
            visual_keywords.extend(
                ["working", "focusing", "concentrating", "determined", "motivated"]
            )
        if any(
            word in text_lower
            for word in ["human", "animal", "species", "evolution"]
        ):
            visual_keywords.extend(
                ["people", "professional", "teamwork", "collaboration"]
            )
        visual_keywords.extend(
            [
                "modern office",
                "workspace",
                "business",
                "success",
                "achievement",
                "technology",
                "digital",
                "innovation",
                "professional",
                "focused",
            ]
        )
        broll_keywords = list(dict.fromkeys(visual_keywords))[:20]
        title = (
            full_text.strip()[:60]
            + ("…" if len(full_text.strip()) > 60 else "")
            if full_text.strip()
            else ""
        )
        description = (
            full_text.strip()[:180]
            + ("…" if len(full_text.strip()) > 180 else "")
            if full_text.strip()
            else ""
        )
        self._print("    🧩 [Heuristics] Meta générées en fallback")
        self._print(
            f"    🔑 Mots-clés B-roll fallback: {', '.join(broll_keywords[:5])}..."
        )
        return title, description, hashtags, broll_keywords
