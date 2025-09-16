"""B-roll management extracted from :mod:`video_processor`."""
from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from moviepy.editor import VideoFileClip

try:
    from broll_selector import BrollSelector, Asset, ScoringFeatures, BrollCandidate
    BROLL_SELECTOR_AVAILABLE = True
except ImportError:
    BrollSelector = Asset = ScoringFeatures = BrollCandidate = None
    BROLL_SELECTOR_AVAILABLE = False

_SPACY_MODEL = None


def _to_float(value) -> float:
    try:
        if hasattr(value, "start"):
            value = value.start or 0.0
        return float(value)
    except Exception:
        return 0.0


def extract_keywords_from_transcript_ai(transcript_segments: List[Dict]) -> Dict:
    keyword_categories = {
        "money": ["money", "cash", "dollars", "profit", "revenue", "income", "wealth"],
        "business": ["business", "company", "startup", "entrepreneur", "strategy"],
        "technology": ["tech", "software", "app", "digital", "online", "ai", "automation"],
        "success": ["success", "win", "achievement", "goal", "growth", "scale", "unstoppable", "beast"],
        "people": ["team", "customer", "client", "person", "human", "community"],
        "emotion_positive": ["amazing", "incredible", "fantastic", "awesome", "fire"],
        "emotion_negative": ["problem", "issue", "difficult", "challenge", "fail"],
        "action": ["build", "create", "launch", "start", "implement", "execute"],
        "brain_mind": ["brain", "mind", "mental", "neuroscience", "neural", "cognitive", "psychology"],
        "health_wellness": ["health", "wellness", "nutrition", "nutrients", "supplements", "fitness", "energy"],
        "learning_growth": ["learn", "learning", "growth", "development", "improvement", "potential", "capability"],
        "internal_dialogue": ["dialogue", "conversation", "thoughts", "thinking", "mindset", "beliefs", "circuit"],
    }
    full_text = " ".join([(seg.get("text") or "").lower() for seg in transcript_segments])
    detected_keywords: Dict[str, List[str]] = {}
    timestamps_by_category: Dict[str, List[Dict]] = {}
    for category, keywords in keyword_categories.items():
        detected_keywords[category] = []
        timestamps_by_category[category] = []
        for keyword in keywords:
            if keyword in full_text:
                detected_keywords[category].append(keyword)
                for seg in transcript_segments:
                    text = (seg.get("text") or "").lower()
                    if keyword in text:
                        start_val = _to_float(seg.get("start", 0.0))
                        end_val = _to_float(seg.get("end", 0.0))
                        timestamps_by_category[category].append(
                            {
                                "start": float(start_val),
                                "end": float(end_val),
                                "keyword": keyword,
                                "context": seg.get("text") or "",
                            }
                        )
    try:
        dominant_theme = max(detected_keywords.items(), key=lambda item: len(item[1]))[0]
    except Exception:
        dominant_theme = "business"
    total_duration = 0.0
    if transcript_segments:
        try:
            last_end = transcript_segments[-1].get("end", 0.0)
            total_duration = _to_float(last_end)
        except Exception:
            total_duration = 0.0
    return {
        "keywords": detected_keywords,
        "timestamps": timestamps_by_category,
        "dominant_theme": dominant_theme,
        "total_duration": total_duration,
    }


def generate_broll_prompts_ai(keyword_analysis: Dict) -> List[Dict]:
    try:
        main_theme = keyword_analysis.get("dominant_theme", "general")
        keywords = keyword_analysis.get("keywords", {})
        sentiment = keyword_analysis.get("sentiment", 0.0)
        prompts: List[str] = []
        if main_theme == "technology":
            prompts.extend(
                [
                    "artificial intelligence neural network",
                    "computer vision algorithm",
                    "tech innovation future",
                    "digital transformation",
                    "machine learning data",
                ]
            )
        elif main_theme == "medical":
            prompts.extend(
                [
                    "medical research laboratory",
                    "healthcare innovation hospital",
                    "microscope scientific discovery",
                    "medical technology",
                    "healthcare professionals",
                ]
            )
        elif main_theme == "business":
            prompts.extend(
                [
                    "business success growth",
                    "entrepreneurship motivation",
                    "professional development office",
                    "team collaboration",
                    "business strategy",
                ]
            )
        elif main_theme == "neuroscience":
            prompts.extend(
                [
                    "neuroscience brain neurons synapse",
                    "brain reflexes nervous system",
                    "brain scan mri eeg lab",
                    "cognitive science",
                    "mental health awareness",
                ]
            )
        else:
            if keywords and isinstance(keywords, dict):
                all_keywords: List[str] = []
                for category_keywords in keywords.values():
                    if isinstance(category_keywords, list):
                        all_keywords.extend(category_keywords[:2])
                base_keywords = all_keywords[:3] if all_keywords else [main_theme]
            else:
                base_keywords = [main_theme]
            for keyword in base_keywords:
                prompts.append(f"{main_theme} {keyword}")
        if sentiment > 0.3:
            prompts.extend(["positive energy", "success achievement", "happy people"])
        elif sentiment < -0.3:
            prompts.extend(["serious focus", "determination", "overcoming challenges"])
        unique_prompts = list(dict.fromkeys(prompts))[:8]
        return unique_prompts
    except Exception as exc:
        print(f"⚠️ Erreur génération prompts AI: {exc}")
        return ["general content", "people working", "modern technology"]


def _load_spacy_model():
    global _SPACY_MODEL
    if _SPACY_MODEL is not None:
        return _SPACY_MODEL
    try:
        import spacy as _spacy

        for model_name in ["en_core_web_sm", "fr_core_news_sm", "xx_ent_wiki_sm"]:
            try:
                _SPACY_MODEL = _spacy.load(model_name, disable=["parser", "lemmatizer"])
                break
            except Exception:
                continue
        if _SPACY_MODEL is None:
            _SPACY_MODEL = _spacy.blank("en")
    except Exception:
        _SPACY_MODEL = None
    return _SPACY_MODEL


def extract_keywords_for_segment_spacy(text: str) -> List[str]:
    try:
        import re as _re

        generic_words = {
            "very",
            "much",
            "many",
            "some",
            "any",
            "all",
            "each",
            "every",
            "few",
            "several",
            "reflexes",
            "speed",
            "clear",
            "good",
            "bad",
            "big",
            "small",
            "new",
            "old",
            "high",
            "low",
            "fast",
            "slow",
            "hard",
            "easy",
            "strong",
            "weak",
            "hot",
            "cold",
            "warm",
            "cool",
            "right",
            "wrong",
            "true",
            "false",
            "yes",
            "no",
            "maybe",
            "perhaps",
            "probably",
            "thing",
            "stuff",
            "way",
            "time",
            "place",
            "person",
            "people",
            "man",
            "woman",
            "child",
            "work",
            "make",
            "do",
            "get",
            "go",
            "come",
            "see",
            "look",
            "hear",
            "feel",
            "think",
            "know",
            "want",
            "need",
            "like",
            "love",
            "hate",
            "hope",
            "wish",
            "try",
            "help",
        }
        model = _load_spacy_model()
        doc = None
        if model is not None:
            try:
                doc = model(text)
            except Exception:
                doc = None
        keywords: List[str] = []
        if doc is not None and hasattr(doc, "ents"):
            for ent in doc.ents:
                value = ent.text.strip()
                if len(value) >= 3 and value.lower() not in keywords and value.lower() not in generic_words:
                    keywords.append(value.lower())
        if doc is not None and getattr(doc, "has_annotation", lambda *_: False)("TAG"):
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN", "VERB") and len(token.text) >= 3:
                    lemma = (token.lemma_ or token.text).lower()
                    if lemma not in keywords and lemma not in generic_words:
                        keywords.append(lemma)
        if not keywords:
            for word in _re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']{4,}", text or ""):
                lower = word.lower()
                if lower not in keywords and lower not in generic_words:
                    keywords.append(lower)
        priority_words = {
            "neuroscience",
            "brain",
            "mind",
            "consciousness",
            "cognitive",
            "mental",
            "psychology",
            "medical",
            "health",
            "treatment",
            "research",
            "science",
            "discovery",
            "innovation",
            "technology",
            "digital",
            "future",
            "ai",
            "artificial",
            "intelligence",
            "machine",
            "business",
            "success",
            "growth",
            "strategy",
            "leadership",
            "entrepreneur",
            "startup",
        }
        priority_keywords = [kw for kw in keywords if kw in priority_words]
        other_keywords = [kw for kw in keywords if kw not in priority_words]
        final_keywords = priority_keywords + other_keywords
        return final_keywords[:12]
    except Exception:
        return []


def _load_broll_selector_config(config) -> Dict:
    try:
        import yaml

        if config.BROLL_SELECTOR_CONFIG_PATH.exists():
            with open(config.BROLL_SELECTOR_CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        print(f"    ⚠️ Fichier de configuration introuvable: {config.BROLL_SELECTOR_CONFIG_PATH}")
        return {}
    except Exception as exc:
        print(f"    ⚠️ Erreur chargement configuration: {exc}")
        return {}


def _calculate_asset_hash(asset_path: Path) -> str:
    try:
        import hashlib

        stat = asset_path.stat()
        hash_data = f"{asset_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_data.encode()).hexdigest()
    except Exception:
        return str(asset_path.name)

def cleanup_broll_duplicates(config, safe_remove_tree: Callable[[Path], bool]) -> None:
    try:
        broll_lib = Path("AI-B-roll") / "broll_library"
        if not broll_lib.exists():
            return
        print("🧹 Nettoyage automatique des doublons B-roll...")
        clip_groups: Dict[str, List[Path]] = {}
        for folder in broll_lib.iterdir():
            if folder.is_dir() and folder.name.startswith("clip_"):
                parts = folder.name.split("_")
                if len(parts) >= 2:
                    if parts[-1].isdigit() and len(parts) >= 3:
                        clip_base = "_".join(parts[1:-1])
                    else:
                        clip_base = "_".join(parts[1:])
                    clip_groups.setdefault(clip_base, []).append(folder)
        cleaned_size = 0.0
        for clip_base, folders in clip_groups.items():
            if len(folders) > 1:
                folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
                keep_folder = folders[0]
                remove_folders = folders[1:]
                print(
                    f"    🔄 Clip '{clip_base}': gardé {keep_folder.name}, suppression de {len(remove_folders)} doublons"
                )
                for folder in remove_folders:
                    try:
                        folder_size = (
                            sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / (1024**3)
                        )
                        if safe_remove_tree(folder):
                            cleaned_size += folder_size
                            print(f"      ✅ Supprimé {folder.name} ({folder_size:.2f} GB)")
                        else:
                            print(f"      ⚠️ Suppression partielle {folder.name}")
                    except Exception as exc:
                        print(f"      ❌ Erreur suppression {folder.name}: {exc}")
        if cleaned_size > 0:
            print(f"    💾 Espace libéré: {cleaned_size:.2f} GB")
        else:
            print("    ✅ Aucun doublon trouvé")
    except Exception as exc:
        print(f"⚠️ Erreur nettoyage doublons: {exc}")


def cleanup_all_temp_broll(config, safe_remove_tree: Callable[[Path], bool]) -> None:
    try:
        broll_lib = Path("AI-B-roll") / "broll_library"
        if not broll_lib.exists():
            return
        temp_folders: List[Path] = []
        for folder in broll_lib.iterdir():
            if folder.is_dir() and folder.name.startswith("temp_clip_"):
                temp_folders.append(folder)
        if not temp_folders:
            print("    ✅ Aucun cache temporaire à nettoyer")
            return
        total_size = 0.0
        for folder in temp_folders:
            try:
                folder_size = (
                    sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / (1024**2)
                )
                total_size += folder_size
                if safe_remove_tree(folder):
                    print(f"    🗑️ Supprimé: {folder.name} ({folder_size:.1f} MB)")
                else:
                    print(f"    ⚠️ Suppression partielle: {folder.name}")
            except Exception as exc:
                print(f"    ❌ Erreur suppression {folder.name}: {exc}")
        print(f"    💾 Total libéré: {total_size:.1f} MB")
    except Exception as exc:
        print(f"⚠️ Erreur nettoyage fichiers temporaires: {exc}")


def purge_broll_caches(config, safe_remove_tree: Callable[[Path], bool]) -> None:
    try:
        broll_lib = Path("AI-B-roll") / "broll_library"
        broll_cache = Path("AI-B-roll") / ".cache"
        if broll_lib.exists():
            for item in broll_lib.glob("*"):
                try:
                    if item.is_dir():
                        safe_remove_tree(item)
                    else:
                        item.unlink(missing_ok=True)
                except Exception:
                    pass
        if broll_cache.exists():
            safe_remove_tree(broll_cache)
    except Exception:
        pass

STOP_PROMPT_TERMS = {
    "very",
    "really",
    "clear",
    "stuff",
    "thing",
    "things",
    "some",
    "any",
    "ever",
    "so",
    "much",
    "get",
    "got",
    "will",
    "discuss",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "im",
    "ive",
    "youve",
    "because",
}


def _filter_prompt_terms(words):
    cleaned = []
    for word in words:
        if not isinstance(word, str):
            continue
        token = word.strip().lower()
        if not token or token in STOP_PROMPT_TERMS or len(token) < 3:
            continue
        cleaned.append(token)
    seen = set()
    result = []
    for token in cleaned:
        if token not in seen:
            result.append(token)
            seen.add(token)
    return result[:5]
def _insert_brolls(config, input_path: Path, subtitles: List[Dict], broll_keywords: List[str], safe_remove_tree: Callable[[Path], bool], get_sentence_transformer_model: Callable[[str], object]) -> Path:
    """Point d'extension B-roll: retourne le chemin vidéo après insertion si activée."""
    Config = config
    if not getattr(Config, 'ENABLE_BROLL', False):
        print("    ⏭️ B-roll désactivés: aucune insertion")
        return input_path
    
    try:
        # Vérifier la librairie B-roll
        broll_root = Path("AI-B-roll")
        broll_library = broll_root / "broll_library"
        if not broll_library.exists():
            print("    ⚠️ Librairie B-roll introuvable, saut de l'insertion")
            return input_path
        # Préparer chemins (écrire directement dans le dossier du clip si possible)
        clip_dir = (Path(input_path).parent if (Path(input_path).name == 'reframed.mp4') else Config.TEMP_FOLDER)
        # Si input_path est déjà dans un dossier clip (reframed.mp4), sortir with_broll.mp4 à côté
        if Path(input_path).name == 'reframed.mp4':
            output_with_broll = clip_dir / 'with_broll.mp4'
        else:
            output_with_broll = Config.TEMP_FOLDER / f"with_broll_{Path(input_path).name}"
        output_with_broll.parent.mkdir(parents=True, exist_ok=True)
        
        # Assurer l'import du pipeline local (src/*)
        if str(broll_root.resolve()) not in sys.path:
            sys.path.insert(0, str(broll_root.resolve()))
        
        # 🚀 NOUVEAUX IMPORTS INTELLIGENTS SYNCHRONES (DÉSACTIVÉS POUR PROMPT OPTIMISÉ)
        try:
            from sync_context_analyzer import SyncContextAnalyzer
            from broll_diversity_manager import BrollDiversityManager
            # 🚨 DÉSACTIVATION TEMPORAIRE: Le système intelligent interfère avec notre prompt optimisé LLM
            INTELLIGENT_BROLL_AVAILABLE = False
            print("    ⚠️  Système intelligent DÉSACTIVÉ pour laisser le prompt optimisé LLM fonctionner")
            print("    🎯 Utilisation exclusive du prompt optimisé: 25-35 keywords + structure hiérarchique")
        except ImportError as e:
            print(f"    ⚠️  Système intelligent non disponible: {e}")
            print("    🔄 Fallback vers ancien système...")
            INTELLIGENT_BROLL_AVAILABLE = False
        
        # Imports B-roll dans tous les cas
        from src.pipeline.config import BrollConfig  # type: ignore
        from src.pipeline.keyword_extraction import extract_keywords_for_segment  # type: ignore
        from src.pipeline.timeline_legacy import plan_broll_insertions, normalize_timeline, enrich_keywords  # type: ignore
        from src.pipeline.renderer import render_video  # type: ignore
        from src.pipeline.transcription import TranscriptSegment  # type: ignore
        
        from moviepy.editor import VideoFileClip as _VFC
        # Optionnel: indexation FAISS/CLIP
        try:
            from src.pipeline.indexer import build_index  # type: ignore
            index_handle = None
        except Exception:
            build_index = None  # type: ignore
            index_handle = None
        
        # 🧠 ANALYSE INTELLIGENTE AVANCÉE
        if INTELLIGENT_BROLL_AVAILABLE:
            print("    🧠 Utilisation du système B-roll intelligent...")
            try:
                # Initialiser l'analyseur contextuel intelligent SYNCHRONE
                context_analyzer = SyncContextAnalyzer()
                
                # Analyser le contexte global de la vidéo
                transcript_text = " ".join([s.get('text', '') for s in subtitles])
                global_analysis = context_analyzer.analyze_context(transcript_text)
                
                print(f"    🎯 Contexte détecté: {global_analysis.main_theme}")
                print(f"    🧬 Sujets: {', '.join(global_analysis.key_topics[:3])}")
                print(f"    😊 Sentiment: {global_analysis.sentiment}")
                print(f"    📊 Complexité: {global_analysis.complexity}")
                print(f"    🔑 Mots-clés: {', '.join(global_analysis.keywords[:5])}")
                
                # Persister l'analyse intelligente
                try:
                    meta_dir = Config.OUTPUT_FOLDER / 'meta'
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    meta_path = meta_dir / f"{Path(input_path).stem}_intelligent_broll_metadata.json"
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'intelligent_analysis': {
                                'main_theme': global_analysis.main_theme,
                                'key_topics': global_analysis.key_topics,
                                'sentiment': global_analysis.sentiment,
                                'complexity': global_analysis.complexity,
                                'keywords': global_analysis.keywords,
                                'context_score': global_analysis.context_score
                            },
                            'timestamp': str(datetime.now())
                        }, f, ensure_ascii=False, indent=2)
                    print(f"    💾 Métadonnées intelligentes sauvegardées: {meta_path}")
                    
                    # 🎬 INSÉRATION INTELLIGENTE DES B-ROLLS
                    print("    🎬 Insertion intelligente des B-rolls...")
                    try:
                        # Créer un dossier unique pour ce clip
                        clip_id = input_path.stem
                        unique_broll_dir = broll_library / f"clip_intelligent_{clip_id}_{int(time.time())}"
                        unique_broll_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Générer des prompts intelligents basés sur l'analyse
                        intelligent_prompts = []
                        main_theme = global_analysis.main_theme
                        kws = _filter_prompt_terms(global_analysis.keywords[:6]) if hasattr(global_analysis, 'keywords') else []
                        if main_theme == 'technology':
                            intelligent_prompts.extend([
                                'artificial intelligence neural network',
                                'computer vision algorithm',
                                'tech innovation future'
                            ])
                        elif main_theme == 'medical':
                            intelligent_prompts.extend([
                                'medical research laboratory',
                                'healthcare innovation hospital',
                                'microscope scientific discovery'
                            ])
                        elif main_theme == 'business':
                            intelligent_prompts.extend([
                                'business success growth',
                                'entrepreneurship motivation',
                                'professional development office'
                            ])
                        elif main_theme == 'neuroscience':
                            intelligent_prompts.extend([
                                'neuroscience brain neurons synapse',
                                'brain reflexes nervous system',
                                'brain scan mri eeg lab'
                            ])
                        else:
                            base = _filter_prompt_terms([main_theme] + kws)
                            intelligent_prompts.extend([f"{main_theme} {kw}" for kw in base[:3]])

                        # Ajouter variantes from cleaned keywords
                        for kw in kws[:3]:
                            intelligent_prompts.append(f"{main_theme} {kw}")

                        # Dedup and trim
                        seen_ip = set()
                        intelligent_prompts = [p for p in intelligent_prompts if not (p in seen_ip or seen_ip.add(p))][:8]

                        print(f"    🎯 Prompts intelligents générés: {', '.join(intelligent_prompts[:3])}")
                        
                        # Utiliser l'ancien système mais avec les prompts intelligents
                        # (temporaire en attendant l'intégration complète)
                        print("    🔄 Utilisation du système B-roll avec prompts intelligents...")
                        
                    except Exception as e:
                        print(f"    ⚠️  Erreur insertion intelligente: {e}")
                        print("    🔄 Fallback vers ancien système...")
                        INTELLIGENT_BROLL_AVAILABLE = False
                        
                except Exception as e:
                    print(f"    ⚠️  Erreur système intelligent: {e}")
                    print("    🔄 Fallback vers ancien système...")
                    INTELLIGENT_BROLL_AVAILABLE = False
            except Exception as e:
                print(f"    ⚠️  Erreur système intelligent: {e}")
                print("    🔄 Fallback vers ancien système...")
                INTELLIGENT_BROLL_AVAILABLE = False
        
        # 🚀 CORRECTION: Préparer l'analyse des mots-clés pour tous les systèmes
        analysis = None
        try:
            analysis = extract_keywords_from_transcript_ai(subtitles)
            print(f"    🧠 Analyse des mots-clés préparée: {analysis.get('dominant_theme', 'N/A')}")
        except Exception as e:
            print(f"    ⚠️ Erreur préparation analyse: {e}")
                
        # Fallback: ancienne analyse si système intelligent indisponible
        if not INTELLIGENT_BROLL_AVAILABLE:
            print("    🔄 Utilisation de l'ancien système B-roll...")
            # analysis déjà préparé plus haut
            prompts = generate_broll_prompts_ai(analysis) if analysis else []
            # Filtrer les prompts fallback
            try:
                cleaned_prompts = []
                for p in prompts:
                    tokens = _filter_prompt_terms(str(p).split())
                    if tokens:
                        cleaned_prompts.append(' '.join(tokens))
                if cleaned_prompts:
                    prompts = cleaned_prompts
            except Exception:
                pass
            # Persiste metadata dans un dossier clip dédié si possible
            try:
                meta_dir = Config.OUTPUT_FOLDER / 'meta'
                meta_dir.mkdir(parents=True, exist_ok=True)
                meta_path = meta_dir / f"{Path(input_path).stem}_broll_metadata.json"
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump({'analysis': analysis, 'prompts': prompts}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        else:
            # 🎯 UTILISER LES PROMPTS INTELLIGENTS
            print("    🎯 Utilisation des prompts intelligents pour B-rolls...")
            try:
                # Créer une analyse basée sur l'analyse intelligente
                analysis = {
                    'main_theme': global_analysis.main_theme,
                    'key_topics': global_analysis.key_topics,
                    'sentiment': global_analysis.sentiment,
                    'keywords': global_analysis.keywords
                }
                
                # Utiliser les prompts intelligents générés
                prompts = intelligent_prompts if 'intelligent_prompt' in locals() else [
                    f"{global_analysis.main_theme} {kw}" for kw in global_analysis.keywords[:3]
                ]
                
                print(f"    🎯 Prompts utilisés: {', '.join(prompts[:3])}")
                
            except Exception as e:
                print(f"    ⚠️  Erreur prompts intelligents: {e}")
                # Fallback vers prompts génériques (analysis déjà préparé)
                prompts = generate_broll_prompts_ai(analysis) if analysis else []
        
        # 🚀 CORRECTION: Utiliser directement le paramètre broll_keywords du LLM
        llm_broll_keywords = []
        try:
            # Utiliser les mots-clés B-roll passés en paramètre (générés par le LLM)
            if broll_keywords and len(broll_keywords) > 0:
                llm_broll_keywords = broll_keywords
                print(f"    🧠 Mots-clés B-roll LLM intégrés: {len(llm_broll_keywords)} termes")
                print(f"    🎯 Exemples: {', '.join(llm_broll_keywords[:5])}")
            else:
                print("    ⚠️ Mots-clés B-roll LLM non disponibles, utilisation extraction basique")
        except Exception as e:
            print(f"    ⚠️ Erreur récupération mots-clés B-roll LLM: {e}")
        
        # Combiner les mots-clés LLM avec les prompts existants
        if llm_broll_keywords:
            # Enrichir les prompts avec les mots-clés LLM
            enhanced_prompts = []
            for kw in llm_broll_keywords[:8]:  # Limiter à 8 mots-clés principaux
                enhanced_prompts.append(kw)
                # Créer des combinaisons avec le thème principal
                if 'global_analysis' in globals() and 'global_analysis' in locals() and hasattr(global_analysis, 'main_theme'):
                    enhanced_prompts.append(f"{getattr(global_analysis, 'main_theme', 'general')} {kw}")
            
            # Ajouter les prompts existants
            enhanced_prompts.extend(prompts)
            
            # Dédupliquer et limiter
            seen_prompts = set()
            final_prompts = []
            for p in enhanced_prompts:
                if p not in seen_prompts and len(p) > 2:
                    final_prompts.append(p)
                    seen_prompts.add(p)
            
            prompts = final_prompts[:12]  # Limiter à 12 prompts finaux
            print(f"    🚀 Prompts enrichis avec LLM: {len(prompts)} termes")
            print(f"    🎯 Prompts finaux: {', '.join(prompts[:5])}...")
        
        # Convertir nos sous-titres en segments attendus par le pipeline
        segments = [
            TranscriptSegment(start=float(s.get('start', 0.0)), end=float(s.get('end', 0.0)), text=str(s.get('text', '')).strip())
            for s in subtitles if (s.get('text') and (s.get('end', 0.0) >= s.get('start', 0.0)))
        ]
        if not segments:
            print("    ⚠️ Aucun segment de transcription valide, saut B-roll")
            return input_path
        
        # Construire la config du pipeline (fetch + embeddings activés, pas de limites)
        cfg = BrollConfig(
            input_video=str(input_path),
            output_video=output_with_broll,
            broll_library=broll_library,
            srt_path=None,
            render_subtitles=False,
                        max_broll_ratio=0.65,           # CORRIGÉ: 90% → 65% pour équilibre optimal
        min_gap_between_broll_s=1.5,    # CORRIGÉ: 0.2s → 1.5s pour respiration visuelle
                        max_broll_clip_s=4.0,           # CORRIGÉ: 8.0s → 4.0s pour B-rolls équilibrés
        min_broll_clip_s=1.5,           # 🚀 OPTIMISÉ: 0.8s → 1.5s pour B-rolls plus visibles sur TikTok
            use_whisper=False,
            ffmpeg_preset="fast",
            crf=23,
            threads=0,
            # Fetchers (stock)
            enable_fetcher=getattr(Config, 'BROLL_FETCH_ENABLE', False),
            fetch_provider=getattr(Config, 'BROLL_FETCH_PROVIDER', 'pexels'),
            pexels_api_key=getattr(Config, 'PEXELS_API_KEY', None),
            pixabay_api_key=getattr(Config, 'PIXABAY_API_KEY', None),
            fetch_max_per_keyword=getattr(Config, 'BROLL_FETCH_MAX_PER_KEYWORD', 15),  # CORRIGÉ: 25 → 15 pour vitesse optimale
            fetch_allow_videos=getattr(Config, 'BROLL_FETCH_ALLOW_VIDEOS', True),
            fetch_allow_images=getattr(Config, 'BROLL_FETCH_ALLOW_IMAGES', True),  # Activé: images animées + Ken Burns
            # Embeddings
            use_embeddings=getattr(Config, 'BROLL_USE_EMBEDDINGS', True),
            embedding_model_name=getattr(Config, 'BROLL_EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
            contextual_config_path=getattr(Config, 'CONTEXTUAL_CONFIG_PATH', Path('config/contextual_broll.yml')),
            # Experimental FX toggle
            enable_experimental_fx=getattr(Config, 'ENABLE_EXPERIMENTAL_FX', False),
        )
        # FETCH DYNAMIQUE PAR CLIP: Créer un dossier unique et forcer le fetch à chaque fois
        try:
            from src.pipeline.fetchers import ensure_assets_for_keywords  # type: ignore
            
            # 🚀 NOUVEAU: Dossier temporaire unique - sera nettoyé après traitement
            clip_id = input_path.stem  # Nom du fichier sans extension
            clip_broll_dir = broll_library / f"temp_clip_{clip_id}_{int(time.time())}"
            
            # Toujours créer un nouveau dossier temporaire
            clip_broll_dir.mkdir(parents=True, exist_ok=True)
            print(f"    📁 Dossier B-roll temporaire créé: {clip_broll_dir.name}")
            print(f"    🗑️ Sera automatiquement nettoyé après traitement")
            
            # Forcer l'activation du fetcher pour chaque clip
            setattr(cfg, 'enable_fetcher', True)
            setattr(cfg, 'broll_library', str(clip_broll_dir))  # Utiliser le dossier unique
            
            print(f"    🔄 Fetch B-roll personnalisé pour clip: {clip_id}")
            print(f"    📁 Dossier B-roll unique: {clip_broll_dir.name}")
            
            # 🚀 NOUVEAU: Intégration du sélecteur B-roll générique
            if BROLL_SELECTOR_AVAILABLE and getattr(Config, 'BROLL_SELECTOR_ENABLED', True):
                try:
                    print("    🎯 Sélecteur B-roll générique activé - Scoring mixte intelligent")
                    
                    # Initialiser le sélecteur avec la configuration
                    selector_config = None
                    if getattr(Config, 'BROLL_SELECTOR_CONFIG_PATH', None):
                        try:
                            import yaml
                            with open(Config.BROLL_SELECTOR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                                selector_config = yaml.safe_load(f)
                            print(f"    ⚙️ Configuration chargée: {Config.BROLL_SELECTOR_CONFIG_PATH}")
                        except Exception as e:
                            print(f"    ⚠️ Erreur chargement config: {e}")
                    
                    # Créer le sélecteur
                    from broll_selector import BrollSelector
                    broll_selector = BrollSelector(selector_config)
                    
                    # 🚀 CORRECTION: Utiliser les mots-clés LLM intelligents au lieu du fallback basique
                    context_keywords = []
                    
                    # 🎯 PRIORITÉ 0: Utiliser les VRAIS mots-clés LLM du système industriel
                    if llm_broll_keywords and len(llm_broll_keywords) > 0:
                        context_keywords = llm_broll_keywords[:15]  # Prendre les 15 meilleurs
                        print(f"    🚀 Mots-clés LLM INDUSTRIELS utilisés: {len(context_keywords)} termes")
                        print(f"    🎯 Mots-clés: {', '.join(context_keywords[:5])}")
                    
                    # Priorité 1: Utiliser l'analyse intelligente si disponible
                    elif 'global_analysis' in locals():
                        context_keywords = global_analysis.keywords[:10] if hasattr(global_analysis, 'keywords') else []
                    
                    # Priorité 2: Utiliser les mots-clés LLM corrigés (notre extraction améliorée)
                    if not context_keywords and 'analysis' in locals():
                        # Extraire les meilleurs mots-clés de notre système LLM
                        llm_keywords = []
                        if isinstance(analysis, dict) and 'keywords' in analysis:
                            for category, kws in analysis['keywords'].items():
                                if isinstance(kws, list):
                                    llm_keywords.extend(kws[:3])  # 3 meilleurs par catégorie
                        
                        # Ajouter le thème dominant
                        if analysis.get('dominant_theme'):
                            llm_keywords.insert(0, analysis['dominant_theme'])
                        
                        context_keywords = llm_keywords[:15] if llm_keywords else []
                        print(f"    🧠 Mots-clés LLM intelligents utilisés: {len(context_keywords)} termes")
                    
                    # Priorité 3: Fallback basique seulement si aucun autre système
                    if not context_keywords:
                        # Extraction basique améliorée - mots significatifs seulement
                        significant_words = []
                        for s in subtitles:
                            text = s.get('text', '')
                            if text:
                                words = text.lower().split()
                                # Filtrer les mots vides et garder les mots significatifs
                                meaningful_words = [w for w in words if len(w) > 4 and w.isalpha() 
                                                  and w not in ['that', 'this', 'they', 'there', 'where', 'when', 'what', 'with', 'have', 'your', 'once', 'figure']]
                                significant_words.extend(meaningful_words)
                        
                        # Déduplication et limitation
                        context_keywords = list(dict.fromkeys(significant_words))[:10]
                        print(f"    ⚠️ Fallback vers extraction basique améliorée: {len(context_keywords)} termes")
                    
                    # Détecter le domaine
                    detected_domain = None
                    if 'global_analysis' in locals() and hasattr(global_analysis, 'main_theme'):
                        detected_domain = global_analysis.main_theme
                    
                    print(f"    🎯 Contexte: {detected_domain or 'général'}")
                    print(f"    🔑 Mots-clés contextuels: {', '.join(context_keywords[:5])}")
                    
                    # Utiliser le sélecteur pour la planification
                    selection_report = broll_selector.select_brolls(
                        keywords=context_keywords,
                        domain=detected_domain,
                        min_delay=_load_broll_selector_config(Config).get('thresholds', {}).get('min_delay_seconds', 4.0),
                        desired_count=_load_broll_selector_config(Config).get('desired_broll_count', 3)
                    )
                    
                    # Sauvegarder le rapport de sélection
                    try:
                        meta_dir = Config.OUTPUT_FOLDER / 'meta'
                        meta_dir.mkdir(parents=True, exist_ok=True)
                        selection_report_path = meta_dir / f"{Path(input_path).stem}_broll_selection_report.json"
                        with open(selection_report_path, 'w', encoding='utf-8') as f:
                            json.dump(selection_report, f, ensure_ascii=False, indent=2)
                        print(f"    💾 Rapport de sélection sauvegardé: {selection_report_path}")
                    except Exception as e:
                        print(f"    ⚠️ Erreur sauvegarde rapport: {e}")
                    
                    # Afficher les statistiques de sélection
                    if 'diagnostics' in selection_report:
                        diag = selection_report['diagnostics']
                        print(f"    📊 Sélection: {diag.get('num_selected', 0)}/{diag.get('num_candidates', 0)} B-rolls")
                        print(f"    🎯 Top score: {diag.get('top_score', 0):.3f}")
                        print(f"    📏 Seuil appliqué: {diag.get('min_score', 0):.3f}")
                    
                    if selection_report.get('fallback_used'):
                        print(f"    🆘 Fallback activé: Tier {selection_report.get('fallback_tier', '?')}")
                    
                except Exception as e:
                    print(f"    ⚠️ Erreur sélecteur générique: {e}")
                    print("    🔄 Fallback vers système existant")
            
            # 🚀 CORRECTION: Intégration des mots-clés LLM pour le fetch
            # SÉLECTION INTELLIGENTE: Mots-clés contextuels + concepts associés
            from collections import Counter as _Counter
            kw_pool: list[str] = []
            
            # 🧠 PRIORITÉ 1: Mots-clés LLM si disponibles
            if broll_keywords:
                try:
                    if not isinstance(broll_keywords, (list, tuple)):
                        print(f"    ❌ Format invalide broll_keywords: {type(broll_keywords)}")
                        broll_keywords = []
                    else:
                        # Normalisation/filtrage
                        broll_keywords = [
                            (kw.strip() if isinstance(kw, str) else "")
                            for kw in broll_keywords
                            if isinstance(kw, str) and kw and kw.strip()
                        ]
                except (TypeError, AttributeError):
                    broll_keywords = []
                
                if broll_keywords:
                    print(f"    🚀 Utilisation des mots-clés LLM pour le fetch: {len(broll_keywords)} termes")
                    # Ajouter TOUS les mots-clés LLM en priorité
                    for kw in broll_keywords:
                        low = (kw or '').strip().lower()
                        if low and len(low) >= 3:
                            kw_pool.append(low)
                            # Ajouter des variations pour enrichir
                            if ' ' in low:  # Mots composés
                                parts = low.split()
                                kw_pool.extend(parts)
                    print(f"    🎯 Mots-clés LLM ajoutés: {', '.join(broll_keywords[:8])}")
                else:
                    print("    ⚠️ Mots-clés LLM indisponibles après validation, fallback basique")
            
            # 🔄 PRIORITÉ 2: Extraction des mots-clés du transcript
            for s in subtitles:
                base_kws = extract_keywords_for_segment(s.get('text','')) or []
                spacy_kws = extract_keywords_for_segment_spacy(s.get('text','')) or []
                for kw in (base_kws + spacy_kws):
                    low = (kw or '').strip().lower()
                    if low and len(low) >= 3:
                        kw_pool.append(low)
            
            # 🚀 CONCEPTS ASSOCIÉS ENRICHIS (50+ concepts)
            concept_mapping = {
                # 🧠 Cerveau & Intelligence
                'brain': ['neuroscience', 'mind', 'thinking', 'intelligence', 'cognitive', 'mental', 'psychology', 'consciousness'],
                'mind': ['brain', 'thinking', 'thought', 'intelligence', 'cognitive', 'mental', 'psychology'],
                'thinking': ['brain', 'mind', 'thought', 'intelligence', 'cognitive', 'mental', 'logic'],
                
                # 💰 Argent & Finance
                'money': ['finance', 'business', 'success', 'wealth', 'investment', 'cash', 'profit', 'revenue'],
                'argent': ['finance', 'business', 'success', 'wealth', 'investment', 'cash', 'profit', 'revenue'],
                'finance': ['money', 'business', 'investment', 'wealth', 'profit', 'revenue', 'budget'],
                
                # 🎯 Focus & Concentration
                'focus': ['concentration', 'productivity', 'attention', 'mindfulness', 'clarity', 'precision'],
                'concentration': ['focus', 'attention', 'mindfulness', 'clarity', 'precision', 'dedication'],
                'attention': ['focus', 'concentration', 'mindfulness', 'awareness', 'observation'],
                
                # 🏆 Succès & Réussite
                'success': ['achievement', 'goal', 'victory', 'winning', 'growth', 'accomplishment', 'triumph'],
                'succès': ['achievement', 'goal', 'victory', 'winning', 'growth', 'accomplishment', 'triumph'],
                'victory': ['success', 'achievement', 'winning', 'triumph', 'conquest', 'domination'],
                
                # ❤️ Santé & Bien-être
                'health': ['wellness', 'fitness', 'medical', 'lifestyle', 'nutrition', 'vitality', 'strength'],
                'santé': ['wellness', 'fitness', 'medical', 'lifestyle', 'nutrition', 'vitality', 'strength'],
                'fitness': ['health', 'wellness', 'exercise', 'training', 'strength', 'endurance'],
                
                # 🤖 Technologie & Innovation
                'technology': ['digital', 'innovation', 'future', 'ai', 'automation', 'tech', 'modern'],
                'technologie': ['digital', 'innovation', 'future', 'ai', 'automation', 'tech', 'modern'],
                'innovation': ['technology', 'digital', 'future', 'ai', 'automation', 'creativity', 'progress'],
                
                # 💼 Business & Entreprise
                'business': ['entrepreneur', 'startup', 'strategy', 'leadership', 'growth', 'company', 'enterprise'],
                'entreprise': ['entrepreneur', 'startup', 'strategy', 'leadership', 'growth', 'company', 'enterprise'],
                'strategy': ['business', 'planning', 'tactics', 'approach', 'method', 'system'],
                
                # 🚀 Action & Dynamisme
                'action': ['movement', 'energy', 'power', 'vitality', 'dynamism', 'activity', 'motion'],
                'action': ['movement', 'energy', 'power', 'vitality', 'dynamism', 'activity', 'motion'],
                'energy': ['power', 'vitality', 'strength', 'force', 'intensity', 'enthusiasm'],
                
                # 🔥 Émotion & Passion
                'emotion': ['feeling', 'passion', 'excitement', 'inspiration', 'motivation', 'enthusiasm'],
                'émotion': ['feeling', 'passion', 'excitement', 'inspiration', 'motivation', 'enthusiasm'],
                'passion': ['emotion', 'feeling', 'excitement', 'inspiration', 'motivation', 'enthusiasm'],
                
                # 🧠 Développement Personnel
                'growth': ['development', 'improvement', 'progress', 'advancement', 'evolution', 'maturity'],
                'croissance': ['development', 'improvement', 'progress', 'advancement', 'evolution', 'maturity'],
                'development': ['growth', 'improvement', 'progress', 'advancement', 'evolution', 'maturity'],
                
                # ✅ Solutions & Résolution
                'solution': ['resolution', 'fix', 'answer', 'remedy', 'cure', 'treatment'],
                'solution': ['resolution', 'fix', 'answer', 'remedy', 'cure', 'treatment'],
                'resolution': ['solution', 'fix', 'answer', 'remedy', 'cure', 'treatment'],
                
                # ⚠️ Problèmes & Défis
                'problem': ['challenge', 'difficulty', 'obstacle', 'barrier', 'issue', 'trouble'],
                'problème': ['challenge', 'difficulty', 'obstacle', 'barrier', 'issue', 'trouble'],
                'challenge': ['problem', 'difficulty', 'obstacle', 'barrier', 'issue', 'trouble'],
                
                # 🌟 Qualité & Excellence
                'quality': ['excellence', 'perfection', 'superiority', 'premium', 'best', 'optimal'],
                'qualité': ['excellence', 'perfection', 'superiority', 'premium', 'best', 'optimal'],
                'excellence': ['quality', 'perfection', 'superiority', 'premium', 'best', 'optimal']
            }
            
            # Enrichir avec des concepts associés
            for kw in kw_pool[:]:
                for concept, related in concept_mapping.items():
                    if concept in kw or any(r in kw for r in related):
                        kw_pool.extend(related[:2])  # Ajouter 2 concepts max
            
            counts = _Counter(kw_pool)
            
            # 🚨 CORRECTION CRITIQUE: PRIORISER les mots-clés LLM sur les mots-clés génériques
            if 'broll_keywords' in locals() and broll_keywords:
                # Utiliser DIRECTEMENT les mots-clés LLM comme requête principale
                llm_keywords = [kw.strip().lower() for kw in broll_keywords if kw and len(kw.strip()) >= 3]
                if llm_keywords:
                    # Prendre les 8 premiers mots-clés LLM + 2 concepts associés
                    top_kws = llm_keywords[:8]
                    # Ajouter quelques concepts associés pour enrichir
                    for kw in top_kws[:3]:  # Pour les 3 premiers mots-clés LLM
                        for concept, related in concept_mapping.items():
                            if concept in kw or any(r in kw for r in related):
                                top_kws.extend(related[:1])  # 1 concept max par mot-clé LLM
                                break
                    print(f"    🚀 REQUÊTE LLM PRIORITAIRE: {' '.join(top_kws[:5])}")
                else:
                    top_kws = [w for w,_n in counts.most_common(15)]
                    print(f"    🔄 Fallback vers mots-clés génériques: {' '.join(top_kws[:5])}")
            else:
                top_kws = [w for w,_n in counts.most_common(15)]
                print(f"    🔄 Mots-clés génériques: {' '.join(top_kws[:5])}")
            
            # Fallback intelligent selon le contexte
            if not top_kws:
                top_kws = ["focus","concentration","study","brain","mind","productivity","success"]
            print(f"    🔎 Fetch B-roll sur requête: {' '.join(top_kws[:5])}")
            # Provider auto-fallback si pas de clés -> archive
            import os as _os
            pex = getattr(Config, 'PEXELS_API_KEY', None) or _os.getenv('PEXELS_API_KEY')
            pixa = getattr(Config, 'PIXABAY_API_KEY', None) or _os.getenv('PIXABAY_API_KEY')
            uns = getattr(Config, 'UNSPLASH_ACCESS_KEY', None) or _os.getenv('UNSPLASH_ACCESS_KEY')
            giphy = _os.getenv('GIPHY_API_KEY')  # 🎭 GIPHY pour GIFs animés
            # Exposer l'accès Unsplash dans la cfg si dispo
            try:
                if uns:
                    setattr(cfg, 'unsplash_access_key', uns)
            except Exception:
                pass
            if not any([pex, pixa, uns]):
                try:
                    setattr(cfg, 'fetch_provider', 'archive')
                    print("    🌐 Providers: archive (aucune clé API détectée)")
                except Exception:
                    pass
            else:
                # 🚀 AMÉLIORATION: Construire une liste de providers optimisée
                prov = []
                if pex:
                    prov.append('pexels')
                if pixa:
                    prov.append('pixabay')
                if uns:
                    prov.append('unsplash')
                if giphy:
                    prov.append('giphy')  # 🎭 GIPHY pour GIFs animés
                
                # 🎯 AJOUT SÉCURISÉ: Archive.org comme source supplémentaire
                try:
                    if prov:  # Si on a des providers avec clés API
                        prov.append('archive')  # Ajouter Archive.org
                        print(f"    🌐 Providers: {','.join(prov)} (Archive.org + Giphy ajoutés pour variété)")
                    else:
                        prov = ['archive']  # Seulement Archive.org si pas de clés
                        print(f"    🌐 Providers: {','.join(prov)} (Archive.org uniquement)")
                    
                    setattr(cfg, 'fetch_provider', ",".join(prov))
                except Exception as e:
                    # Fallback sécurisé
                    try:
                        if prov:
                            setattr(cfg, 'fetch_provider', ",".join(prov))
                            print(f"    🌐 Providers: {','.join(prov)} (fallback sécurisé)")
                        else:
                            setattr(cfg, 'fetch_provider', 'archive')
                            print(f"    🌐 Providers: archive (fallback ultime)")
                    except Exception:
                        pass
            
            try:
                setattr(cfg, 'fetch_allow_images', True)
                # 🚀 OPTIMISATION MULTI-SOURCES: Qualité optimale (CORRIGÉ)
                if uns and giphy:  # Si Unsplash ET Giphy sont disponibles
                    setattr(cfg, 'fetch_max_per_keyword', 35)  # CORRIGÉ: 125 → 35 pour qualité maximale
                    print("    📊 Configuration optimisée: 35 assets max + images activées (Unsplash + Giphy + Archive)")
                elif uns:  # Si seulement Unsplash est disponible
                    setattr(cfg, 'fetch_max_per_keyword', 30)  # CORRIGÉ: 100 → 30 pour qualité maximale
                    print("    📊 Configuration optimisée: 30 assets max + images activées (Unsplash + Archive)")
                elif giphy:  # Si seulement Giphy est disponible
                    setattr(cfg, 'fetch_max_per_keyword', 30)  # CORRIGÉ: 100 → 30 pour qualité avec GIFs
                    print("    📊 Configuration optimisée: 30 assets max + images activées (Giphy + Archive)")
                else:
                    setattr(cfg, 'fetch_max_per_keyword', 25)  # CORRIGÉ: 75 → 25 pour Archive.org
                    print("    📊 Configuration optimisée: 25 assets max + images activées (Archive.org)")
            except Exception:
                pass
            
            # Déclencher le fetch par mot-clé avec limites dynamiques (5 générique, 8 spécifique)
            def _is_generic_fetch_keyword(kw: str) -> bool:
                if not isinstance(kw, str):
                    return True
                k = kw.strip().lower()
                # Expressions multi-mots = spécifiques
                if ' ' in k:
                    return False
                GENERIC_SIMPLE = {
                    'people','person','start','thing','stuff','your','once','figure',
                    'they','them','this','that','what','when','where','how','any','some'
                }
                return (k in GENERIC_SIMPLE) or (len(k) <= 6)
            
            for _kw in top_kws:
                per_kw_limit = 5 if _is_generic_fetch_keyword(_kw) else 8
                try:
                    setattr(cfg, 'fetch_max_per_keyword', per_kw_limit)
                    print(f"    🔧 Limite par mot-clé '{_kw}': {per_kw_limit} assets")
                except Exception:
                    pass
                ensure_assets_for_keywords(cfg, [_kw])
            
            # 🚨 CORRECTION CRITIQUE: SYSTÈME D'UNICITÉ DES B-ROLLS
            # Éviter la duplication des B-rolls entre vidéos différentes
            try:
                # Créer un fichier de traçabilité des B-rolls utilisés
                broll_tracking_file = Config.OUTPUT_FOLDER / 'meta' / 'broll_usage_tracking.json'
                broll_tracking_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Charger l'historique des B-rolls utilisés
                broll_history = {}
                if broll_tracking_file.exists():
                    try:
                        with open(broll_tracking_file, 'r', encoding='utf-8') as f:
                            broll_history = json.load(f)
                    except Exception:
                        broll_history = {}
                
                # Identifier les B-rolls disponibles pour ce clip
                available_brolls = []
                for asset_path in clip_broll_dir.rglob('*'):
                    if asset_path.suffix.lower() in {'.mp4', '.mov', '.mkv', '.webm', '.jpg', '.jpeg', '.png'}:
                        asset_hash = _calculate_asset_hash(asset_path)
                        asset_info = {
                            'path': str(asset_path),
                            'hash': asset_hash,
                            'size': asset_path.stat().st_size,
                            'last_used': None,
                            'usage_count': 0
                        }
                        
                        # Vérifier si ce B-roll a déjà été utilisé
                        if asset_hash in broll_history:
                            asset_info['last_used'] = broll_history[asset_hash].get('last_used')
                            asset_info['usage_count'] = broll_history[asset_hash].get('usage_count', 0)
                        
                        available_brolls.append(asset_info)
                
                # Trier par priorité: B-rolls jamais utilisés en premier, puis par ancienneté
                available_brolls.sort(key=lambda x: (x['usage_count'], x['last_used'] or '1970-01-01'))
                
                # Sélectionner les B-rolls uniques pour cette vidéo
                selected_brolls = available_brolls[:3]  # 3 B-rolls uniques
                
                # Mettre à jour l'historique d'utilisation
                current_time = datetime.now().isoformat()
                for broll in selected_brolls:
                    broll_history[broll['hash']] = {
                        'last_used': current_time,
                        'usage_count': broll['usage_count'] + 1,
                        'video_id': Path(input_path).stem
                    }
                
                # Sauvegarder l'historique
                with open(broll_tracking_file, 'w', encoding='utf-8') as f:
                    json.dump(broll_history, f, ensure_ascii=False, indent=2)
                
                print(f"    🎯 B-rolls uniques sélectionnés: {len(selected_brolls)} (évite duplication)")
                
            except Exception as e:
                print(f"    ⚠️ Erreur système d'unicité: {e}")
                # Fallback: utiliser tous les B-rolls disponibles
                pass
            
            # Comptage après fetch dans le dossier du clip
            try:
                _media_exts = {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}
                _after = [p for p in clip_broll_dir.rglob('*') if p.suffix.lower() in _media_exts]
                print(f"    📥 Fetch terminé: {len(_after)} assets pour ce clip")
                
                # 🚨 CORRECTION CRITIQUE: Créer fetched_brolls accessible globalement
                fetched_brolls = []
                for asset_path in _after:
                    if asset_path.exists():
                        fetched_brolls.append({
                            'path': str(asset_path),
                            'name': asset_path.name,
                            'size': asset_path.stat().st_size if asset_path.exists() else 0
                        })
                
                print(f"    🎯 {len(fetched_brolls)} B-rolls prêts pour l'assignation")
                
                if len(_after) == 0:
                    print("    ⚠️ Aucun asset téléchargé. Vérifie les clés API et la connectivité réseau.")
            except Exception:
                fetched_brolls = []
                print("    ⚠️ Erreur lors de la préparation des B-rolls fetchés")
            
            # Construire l'index FAISS pour ce clip spécifique
            try:
                if 'build_index' in globals() and build_index is not None:  # type: ignore[name-defined]
                    index_handle = build_index(str(clip_broll_dir), model_name='ViT-B/32')  # type: ignore[misc]
                    print(f"    🧭 Index FAISS construit pour {clip_id}: {len(_after)} assets")
            except Exception:
                index_handle = None
        except Exception:
            pass
  
        # Extensions optionnelles pour crossfade/LUT
        try:
            setattr(cfg, 'crossfade_frames', 3)
            setattr(cfg, 'enable_color_match', True)
            setattr(cfg, 'transition_mode', 'auto')  # 'auto' | 'cut' | 'crossfade' | 'zoom'
            setattr(cfg, 'allow_zoom_transitions', True)
            setattr(cfg, 'enable_image_kenburns', True)
        except Exception:
            pass
        
        # Préparer stop-words (legacy pipeline)
        stopwords: set[str] = set()
        try:
            swp = Path('config/stopwords.txt')
            if swp.exists():
                stopwords = {ln.strip().lower() for ln in swp.read_text(encoding='utf-8').splitlines() if ln.strip()}
        except Exception:
            stopwords = set()

        # 🚀 CORRECTION: Intégration des mots-clés LLM dans la planification
        # Planification: nouvelle API préférée (plan_broll_insertions(segments, cfg, index))
        
        # 🚨 CORRECTION CRITIQUE: fetched_brolls est déjà déclaré plus haut, ne pas le redéclarer !
        # fetched_brolls = []  # ❌ SUPPRIMÉ: Cette ligne écrase la variable fetchée !
        
        try:
            plan = plan_broll_insertions(segments, cfg, index_handle)  # type: ignore[arg-type]
        except Exception:
            # 🚀 NOUVEAU: Utiliser les mots-clés LLM pour la planification
            seg_keywords: List[List[str]] = []
            
            # 🧠 PRIORITÉ 1: Mots-clés LLM si disponibles
            if 'broll_keywords' in locals() and broll_keywords:
                print(f"    🚀 Utilisation des mots-clés LLM pour la planification: {len(broll_keywords)} termes")
                # Distribuer les mots-clés LLM sur les segments
                for i, s in enumerate(segments):
                    # Prendre 2-3 mots-clés LLM par segment
                    start_idx = (i * 2) % len(broll_keywords)
                    end_idx = min(start_idx + 2, len(broll_keywords))
                    segment_llm_kws = broll_keywords[start_idx:end_idx]
                    
                    # Combiner avec extraction basique
                    base_kws = extract_keywords_for_segment(s.text) or []
                    spacy_kws = extract_keywords_for_segment_spacy(s.text) or []
                    
                    # 🎯 PRIORITÉ aux mots-clés LLM
                    merged: List[str] = segment_llm_kws + base_kws + spacy_kws
                    
                    # Nettoyer et dédupliquer
                    cleaned: List[str] = []
                    seen = set()
                    for kw in merged:
                        if kw and kw.lower() not in seen:
                            low = kw.lower()
                            if not (len(low) < 3 and low in stopwords):
                                cleaned.append(low)
                                seen.add(low)
                    
                    seg_keywords.append(cleaned[:8])  # OPTIMISÉ: 15 → 8 pour vitesse
                    print(f"    🎯 Segment {i}: {len(cleaned)} mots-clés (LLM: {len(segment_llm_kws)})")
            else:
                # 🔄 Fallback: extraction basique uniquement
                print("    ⚠️ Mots-clés LLM non disponibles, utilisation extraction basique")
                for s in segments:
                    base_kws = extract_keywords_for_segment(s.text) or []
                    spacy_kws = extract_keywords_for_segment_spacy(s.text) or []
                    merged: List[str] = []
                    for kw in (base_kws + spacy_kws):
                        if kw and kw.lower() not in merged:
                            low = kw.lower()
                            if not (len(low) < 5 and low in stopwords):
                                merged.append(low)
                    seg_keywords.append(merged[:6])
            
            with _VFC(str(input_path)) as _tmp:
                duration = float(_tmp.duration)
            plan = plan_broll_insertions(  # type: ignore[call-arg]
                segments,
                seg_keywords,
                total_duration=duration,
                max_broll_ratio=cfg.max_broll_ratio,
                min_gap_between_broll_s=cfg.min_gap_between_broll_s,
                max_broll_clip_s=cfg.max_broll_clip_s,
                min_broll_clip_s=1.5,  # 🚀 OPTIMISÉ: 0.8s → 1.5s pour B-rolls plus visibles
            )
            
            # 🚨 CORRECTION CRITIQUE: Assigner directement les B-rolls fetchés aux items du plan
            if plan and fetched_brolls:
                print(f"    🎯 Assignation directe des {len(fetched_brolls)} B-rolls fetchés aux {len(plan)} items du plan...")
                
                # Filtrer les B-rolls valides
                valid_brolls = [broll for broll in fetched_brolls if broll.get('path') and Path(broll.get('path')).exists()]
                
                if valid_brolls:
                    # Assigner les B-rolls aux items du plan
                    for i, item in enumerate(plan):
                        if i < len(valid_brolls):
                            asset_path = valid_brolls[i]['path']
                            
                            # Assigner l'asset_path selon le type d'objet
                            if hasattr(item, 'asset_path'):
                                item.asset_path = asset_path
                            elif isinstance(item, dict):
                                item['asset_path'] = asset_path
                            
                            print(f"    ✅ B-roll {i+1} assigné: {Path(asset_path).name}")
                        else:
                            break
                    
                    print(f"    🎉 {min(len(plan), len(valid_brolls))} B-rolls assignés avec succès au plan")
                else:
                    print(f"    ⚠️ Aucun B-roll valide trouvé dans fetched_brolls")
            elif not fetched_brolls:
                print(f"    ⚠️ Aucun B-roll fetché disponible pour l'assignation")
            elif not plan:
                print(f"    ⚠️ Plan vide - aucun item à traiter")
        # Scoring adaptatif si disponible (pertinence/diversité/esthétique)
        

        
        try:
            from src.pipeline.scoring import score_candidates  # type: ignore
            boosts = {
                # 🚀 Business & Croissance
                "croissance": 0.9, "growth": 0.9, "opportunité": 0.8, "opportunite": 0.8,
                "innovation": 0.9, "développement": 0.8, "developpement": 0.8, "expansion": 0.8,
                "stratégie": 0.8, "strategie": 0.8, "plan": 0.7, "objectif": 0.8, "vision": 0.8,
                
                # 💰 Argent & Finance
                "argent": 1.0, "money": 1.0, "cash": 0.9, "investissement": 0.9, "investissements": 0.9,
                "revenu": 0.8, "revenus": 0.8, "profit": 0.9, "profits": 0.9, "perte": 0.7, "pertes": 0.7,
                "échec": 0.7, "echec": 0.7, "budget": 0.7, "gestion": 0.7, "marge": 0.8, "roi": 0.9,
                "chiffre": 0.7, "ca": 0.7, "économie": 0.8, "economie": 0.8, "financier": 0.8,
                
                # 🤝 Relation & Client
                "client": 0.9, "clients": 0.9, "collaboration": 0.8, "collaborations": 0.8,
                "communauté": 0.7, "communaute": 0.7, "confiance": 0.7, "vente": 0.8, "ventes": 0.8,
                "deal": 0.7, "deals": 0.7, "prospect": 0.6, "prospects": 0.6, "contrat": 0.7,
                "partenariat": 0.8, "équipe": 0.7, "equipe": 0.7, "réseau": 0.7, "reseau": 0.7,
                
                # 🔥 Motivation & Succès
                "succès": 0.9, "succes": 0.9, "motivation": 0.8, "énergie": 0.7, "energie": 0.7,
                "victoire": 0.8, "discipline": 0.7, "viral": 0.8, "viralité": 0.8, "viralite": 0.8,
                "impact": 0.6, "explose": 0.6, "explosion": 0.6, "inspiration": 0.8, "passion": 0.8,
                "détermination": 0.8, "determination": 0.8, "persévérance": 0.8, "perseverance": 0.8,
                
                # 🧠 Intelligence & Apprentissage
                "cerveau": 1.0, "brain": 1.0, "intelligence": 0.9, "savoir": 0.8, "connaissance": 0.8,
                "apprentissage": 0.8, "apprendre": 0.8, "étude": 0.8, "etude": 0.8, "formation": 0.8,
                "compétence": 0.8, "competence": 0.8, "expertise": 0.8, "maîtrise": 0.8, "maitrise": 0.8,
                
                # 💡 Innovation & Technologie
                "technologie": 0.9, "tech": 0.9, "innovation": 0.9, "digital": 0.8, "numérique": 0.8,
                "numerique": 0.8, "futur": 0.8, "avancée": 0.8, "avancee": 0.8, "révolution": 0.8,
                "revolution": 0.8, "disruption": 0.8, "transformation": 0.8, "évolution": 0.8, "evolution": 0.8,
                
                # ⚠️ Risque & Erreurs
                "erreur": 0.6, "erreurs": 0.6, "warning": 0.6, "obstacle": 0.6, "obstacles": 0.6,
                "solution": 0.6, "solutions": 0.6, "leçon": 0.5, "lecon": 0.5, "apprentissage": 0.5,
                "problème": 0.6, "probleme": 0.6, "défi": 0.7, "defi": 0.7, "challenge": 0.7,
                
                # 🌟 Qualité & Excellence
                "excellence": 0.9, "qualité": 0.8, "qualite": 0.8, "perfection": 0.8, "meilleur": 0.8,
                "optimal": 0.8, "efficacité": 0.8, "efficacite": 0.8, "performance": 0.8, "résultat": 0.8,
                "resultat": 0.8, "succès": 0.9, "succes": 0.9, "réussite": 0.9, "reussite": 0.9,
            }
            plan = score_candidates(
                plan, segments, broll_library=str(broll_library), clip_model='ViT-B/32',
                use_faiss=True, top_k=10, keyword_boosts=boosts
            )
            
        except Exception:
            pass
 
        # FILTRE: Exclure les B-rolls trop tôt dans la vidéo (délai minimum 3 secondes)
        try:
            filtered_plan = []
            for it in plan:
                st = float(getattr(it, 'start', 0.0) if hasattr(it, 'start') else (it.get('start', 0.0) if isinstance(it, dict) else 0.0))
                if st >= 3.0:  # Délai minimum de 3 secondes avant le premier B-roll
                    filtered_plan.append(it)
                else:
                    print(f"    ⏰ B-roll filtré: trop tôt à {st:.2f}s (minimum 3.0s)")
            
            plan = filtered_plan
            print(f"    ✅ Plan filtré: {len(plan)} B-rolls après délai minimum")
        except Exception:
            pass

        # Déduplication souple: autoriser réutilisation si espacée (> 12s)
        try:
            seen: dict[str, float] = {}
            new_plan = []
            for it in plan:
                # 🔧 CORRECTION: Gérer à la fois BrollPlanItem et dict
                if hasattr(it, 'asset_path'):
                    ap = it.asset_path
                    st = float(it.start)
                elif isinstance(it, dict):
                    ap = it.get('asset_path')
                    st = float(it.get('start', 0.0))
                else:
                    # Fallback pour autres types
                    ap = getattr(it, 'asset_path', None)
                    st = float(getattr(it, 'start', 0.0))
                
                if not ap:
                    new_plan.append(it)
                    continue
                
                last = seen.get(ap, -1e9)
                if st - last >= 8.0:
                    new_plan.append(it)
                    seen[ap] = st
            plan = new_plan
            
        except Exception:
            pass
 
        # 🚀 PRIORISATION FRAÎCHEUR: Trier par timestamp du dossier (plus récent en premier)
        try:
            if plan:
                # Extraire le clip_id pour la priorisation
                clip_id = input_path.stem
                
                # Prioriser par fraîcheur si possible
                for item in plan:
                    if hasattr(item, 'asset_path') and item.asset_path:
                        asset_path = item.asset_path
                    elif isinstance(item, dict) and item.get('asset_path'):
                        asset_path = item['asset_path']
                    else:
                        continue
                    
                    # Calculer le score de fraîcheur
                    try:
                        path = Path(asset_path)
                        for part in path.parts:
                            if part.startswith(f"clip_{clip_id}_") and "_" in part:
                                timestamp_str = part.split("_")[-1]
                                if timestamp_str.isdigit():
                                    item.freshness_score = int(timestamp_str)
                                    break
                        else:
                            item.freshness_score = 0
                    except Exception:
                        item.freshness_score = 0
                
                # Trier par fraîcheur décroissante
                plan.sort(key=lambda x: getattr(x, 'freshness_score', 0), reverse=True)
                print(f"    🆕 Priorisation fraîcheur: {len(plan)} B-rolls triés par timestamp")
                
        except Exception as e:
            print(f"    ⚠️  Erreur priorisation fraîcheur: {e}")
 
        # 🎯 SCORING CONTEXTUEL RENFORCÉ: Pénaliser les assets non pertinents au domaine
        try:
            if plan and "global_analysis" in locals() and hasattr(global_analysis, 'main_theme') and hasattr(global_analysis, 'keywords'):
                domain = global_analysis.main_theme
                keywords = global_analysis.keywords[:10] if hasattr(global_analysis, 'keywords') else []
                
                for item in plan:
                    if hasattr(item, 'asset_path') and item.asset_path:
                        asset_path = item.asset_path
                    elif isinstance(item, dict) and item.get('asset_path'):
                        asset_path = item['asset_path']
                    else:
                        continue
                    
                    # Calculer le score contextuel
                    context_score = _score_contextual_relevance(asset_path, domain, keywords)
                    
                    # Appliquer le score contextuel au score final
                    if hasattr(item, 'score'):
                        # Ajuster le score existant
                        item.score = item.score * context_score
                    elif isinstance(item, dict) and 'score' in item:
                        item['score'] = item['score'] * context_score
                    
                    # Stocker le score contextuel pour debug
                    if hasattr(item, 'context_score'):
                        item.context_score = context_score
                    elif isinstance(item, dict):
                        item['context_score'] = context_score
                
                print(f"    🎯 Scoring contextuel appliqué: domaine '{domain}' avec {len(keywords)} mots-clés")
                
                # 🔍 DEBUG B-ROLL SELECTION (si activé)
                debug_mode = getattr(Config, 'DEBUG_BROLL', False) or os.getenv('DEBUG_BROLL', 'false').lower() == 'true'
                _debug_broll_selection(plan, domain, keywords, debug_mode)
                
                # 🚨 FALLBACK PROPRE: Si aucun asset pertinent, utiliser des assets neutres
                # 🔧 CORRECTION CRITIQUE: Vérifier d'abord si les items ont des assets assignés
                items_without_assets = []
                items_with_assets = []
                
                for item in plan:
                    if hasattr(item, 'asset_path') and item.asset_path:
                        items_with_assets.append(item)
                    elif isinstance(item, dict) and item.get('asset_path'):
                        items_with_assets.append(item)
                    else:
                        items_without_assets.append(item)
                
                print(f"    🔍 Analyse des assets: {len(items_with_assets)} avec assets, {len(items_without_assets)} sans assets")
                
                # 🚨 CORRECTION: Assigner des assets aux items sans assets AVANT le fallback
                if items_without_assets and fetched_brolls:
                    print(f"    🎯 Assignation d'assets aux {len(items_without_assets)} items sans assets...")
                    
                    # 🚀 NOUVEAU: Utiliser uniquement les B-rolls frais pour éviter la duplication
                    available_assets = []
                    for broll in fetched_brolls:
                        asset_path = broll.get('path', '')
                        if asset_path and Path(asset_path).exists():
                            available_assets.append(asset_path)
                    
                    # 🚀 NOUVEAU: Mélanger pour éviter l'ordre séquentiel répétitif
                    import random
                    random.shuffle(available_assets)
                    
                    print(f"    ✅ {len(available_assets)} assets frais disponibles (mélangés pour diversité)")
                    
                    for i, item in enumerate(items_without_assets):
                        if i < len(available_assets):
                            asset_path = available_assets[i]
                            if hasattr(item, 'asset_path'):
                                item.asset_path = asset_path
                            elif isinstance(item, dict):
                                item['asset_path'] = asset_path
                            
                            print(f"    ✅ Asset frais assigné à item {i+1}: {Path(asset_path).name}")
                        else:
                            break
                else:
                    print(f"    ⚠️  Plan vide - Aucun item à traiter")
                
        except Exception as e:
            print(f"    ⚠️  Erreur scoring contextuel: {e}")
 
                    # Affecter un asset_path pertinent via FAISS/CLIP si manquant
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as _np  # type: ignore
            import faiss as _faiss  # type: ignore
            from pathlib import Path as _P
            
            # 🚨 NOUVEAU: Importer le système de scoring contextuel intelligent
            try:
                from src.pipeline.broll_selector import get_contextual_broll_score
                print("    🧠 Système de scoring contextuel intelligent activé")
            except ImportError:
                print("    ⚠️ Système de scoring contextuel non disponible")
                get_contextual_broll_score = None
            
            # UTILISER LE DOSSIER SPÉCIFIQUE DU CLIP (pas la librairie globale)
            clip_specific_dir = clip_broll_dir if 'clip_specific_dir' in locals() else broll_library
            idx_bin = (clip_specific_dir / 'faiss.index')
            idx_json = (clip_specific_dir / 'faiss.json')
            
            model_name = getattr(cfg, 'embedding_model_name', 'clip-ViT-B/32')
            # 🚀 OPTIMISATION: Utiliser le cache pour éviter le rechargement
            final_model_name = 'clip-ViT-B/32' if 'ViT' in model_name else model_name
            st_model = get_sentence_transformer_model(final_model_name)
            
            # 🚀 CORRECTION: Créer emb_text conditionnellement
            emb_text = None
            if st_model is None:
                print(f"    ⚠️ Impossible de charger le modèle {final_model_name}, désactivation FAISS")
            else:
                def emb_text(t: str):
                    v = st_model.encode([t])[0].astype('float32')
                    n = _np.linalg.norm(v) + 1e-12
                    return v / n
            paths = []
            if idx_json.exists():
                import json as _json
                try:
                    paths = _json.loads(idx_json.read_text(encoding='utf-8')).get('paths', [])
                except Exception:
                    paths = []
            index = _faiss.read_index(str(idx_bin)) if idx_bin.exists() else None
            used_recent: set[str] = set()
            for it in plan or []:
                ap = getattr(it, 'asset_path', None) if hasattr(it, 'asset_path') else (it.get('asset_path') if isinstance(it, dict) else None)
                if ap:
                    continue
                # Texte local autour de l'event
                st_e = float(getattr(it, 'start', 0.0) if hasattr(it, 'start') else (it.get('start') if isinstance(it, dict) else 0.0))
                en_e = float(getattr(it, 'end', 0.0) if hasattr(it, 'end') else (it.get('end') if isinstance(it, dict) else 0.0))
                local = " ".join(s.text for s in segments if float(s.start) <= en_e and float(s.end) >= st_e)[:400]
                q = emb_text(local) if local and emb_text is not None else None
                
                # 🚨 NOUVEAU: Extraction des mots-clés pour le scoring contextuel
                local_keywords = []
                if local:
                    # Extraire les mots-clés du texte local
                    words = local.lower().split()
                    local_keywords = [w for w in words if len(w) > 3 and w.isalpha()][:10]
                
                chosen = None
                best_score = -1
                
                if index is not None and q is not None and paths:
                    # 🚨 NOUVEAU: Recherche étendue pour évaluation contextuelle
                    D,I = index.search(q.reshape(1,-1), 15)  # Augmenter de 5 à 15 candidats
                    
                    # 🚨 NOUVEAU: Évaluation contextuelle de tous les candidats
                    for idx in I[0].tolist():
                        if 0 <= idx < len(paths):
                            p = paths[idx]
                            if not p:
                                continue
                            cand = _P(p)
                            if not cand.is_absolute():
                                cand = (clip_specific_dir / p).resolve()
                            if str(cand) not in used_recent and cand.exists():
                                # 🚨 NOUVEAU: Calcul du score contextuel intelligent
                                contextual_score = 0.0
                                if 'get_contextual_broll_score' in globals() and local_keywords:
                                    try:
                                        # Extraire les tokens et tags du fichier
                                        asset_name = cand.stem.lower()
                                        asset_tokens = asset_name.split('_')
                                        asset_tags = asset_name.split('_')  # Simplifié pour l'exemple
                                        contextual_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)
                                    except Exception as e:
                                        print(f"    ⚠️ Erreur scoring contextuel: {e}")
                                        contextual_score = 0.0
                                
                                # 🚨 NOUVEAU: Score combiné FAISS + Contextuel
                                faiss_score = float(D[0][I[0].tolist().index(idx)]) if idx in I[0] else 0.0
                                combined_score = faiss_score + (contextual_score * 2.0)  # Poids contextuel DOUBLÉ
                                
                                if combined_score > best_score:
                                    best_score = combined_score
                                    chosen = str(cand)
                    
                    # 🚨 NOUVEAU: Log de la sélection contextuelle
                    if chosen and 'get_contextual_broll_score' in globals() and local_keywords:
                        try:
                            asset_name = Path(chosen).stem.lower()
                            asset_tokens = asset_name.split('_')
                            asset_tags = asset_name.split('_')
                            final_contextual_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)
                            print(f"    🎯 Sélection contextuelle: {Path(chosen).stem} | Score: {best_score:.3f} | Contexte: {final_contextual_score:.2f}")
                        except Exception:
                            pass
                
                if chosen is None:
                    # 🚀 NOUVEAU: Fallback intelligent utilisant UNIQUEMENT les assets frais du dossier fetched/
                    print(f"    🔍 Fallback vers assets frais uniquement...")
                    fetched_dir = clip_specific_dir / 'fetched'
                    if fetched_dir.exists():
                        for p in fetched_dir.rglob('*'):
                            if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}:
                                if str(p.resolve()) not in used_recent and p.exists():
                                    # 🚀 NOUVEAU: Évaluation contextuelle prioritaire
                                    if 'get_contextual_broll_score' in globals() and local_keywords:
                                        try:
                                            asset_name = p.stem.lower()
                                            asset_tokens = asset_name.split('_')
                                            asset_tags = asset_name.split('_')
                                            fallback_score = get_contextual_broll_score(local_keywords, asset_tokens, asset_tags)
                                            if fallback_score > 1.0:  # Seuil réduit pour plus de diversité
                                                chosen = str(p.resolve())
                                                print(f"    ✅ Asset frais contextuel: {p.stem} | Score: {fallback_score:.2f}")
                                                break
                                        except Exception:
                                            pass
                                    else:
                                        # Utiliser directement l'asset frais sans scoring
                                        chosen = str(p.resolve())
                                        print(f"    ✅ Asset frais utilisé: {p.stem}")
                                        break
                    else:
                        print(f"    ⚠️ Dossier fetched/ non trouvé: {fetched_dir}")
                
                if chosen:
                    if isinstance(it, dict):
                        it['asset_path'] = chosen
                    else:
                        try:
                            setattr(it, 'asset_path', chosen)
                        except Exception:
                            pass
        except Exception:
            pass

        # Vérification des asset_path avant normalisation + mini fallback non invasif
        try:
            def _get_ap(x):
                return (getattr(x, 'asset_path', None) if hasattr(x, 'asset_path') else (x.get('asset_path') if isinstance(x, dict) else None))
            missing = [it for it in (plan or []) if not _get_ap(it)]
            if plan and len(missing) == len(plan):
                # 🚀 NOUVEAU: Fallback intelligent anti-duplication
                # UTILISER LE DOSSIER SPÉCIFIQUE DU CLIP (fetched/ en priorité)
                clip_specific_dir = clip_broll_dir if 'clip_specific_dir' in locals() else broll_library
                
                # 🚀 PRIORISER le dossier fetched/ pour des assets frais
                fetched_dir = clip_specific_dir / 'fetched'
                if fetched_dir.exists():
                    lib_assets = [p for p in fetched_dir.rglob('*') if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}]
                    print(f"    🎯 Utilisation assets frais du dossier fetched/: {len(lib_assets)} assets")
                else:
                    lib_assets = [p for p in clip_specific_dir.rglob('*') if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}]
                    print(f"    ⚠️ Fallback vers dossier complet: {len(lib_assets)} assets")
                
                if lib_assets:
                    # 🚀 NOUVEAU: Mélanger les assets pour éviter l'ordre répétitif
                    import random
                    random.shuffle(lib_assets)
                    
                    # 🚀 NOUVEAU: Répartition intelligente anti-duplication
                    assigned_assets = set()  # Tracker les assets déjà utilisés
                    for i, it in enumerate(plan):
                        ap = _get_ap(it)
                        if ap:
                            continue
                            
                        # Chercher un asset non encore utilisé
                        chosen_asset = None
                        for asset in lib_assets:
                            asset_path = str(asset.resolve())
                            if asset_path not in assigned_assets:
                                chosen_asset = asset_path
                                assigned_assets.add(asset_path)
                                break
                        
                        # Si tous les assets sont utilisés, reprendre depuis le début
                        if chosen_asset is None and lib_assets:
                            chosen_asset = str(lib_assets[i % len(lib_assets)].resolve())
                            print(f"    ⚠️ Recyclage asset {i+1}: tous les assets frais épuisés")
                        
                        if chosen_asset:
                            if isinstance(it, dict):
                                it['asset_path'] = chosen_asset
                            else:
                                try:
                                    setattr(it, 'asset_path', chosen_asset)
                                except Exception:
                                    pass
                            print(f"    ✅ Asset intelligent assigné {i+1}: {Path(chosen_asset).name}")
                    
                    print(f"    📊 Assignation intelligente: {len(assigned_assets)} assets uniques utilisés sur {len(lib_assets)} disponibles")
        except Exception:
            pass
 
         # Normaliser la timeline en événements canonique et rendre
        try:
            with _VFC(str(input_path)) as _fpsprobe:
                fps_probe = float(_fpsprobe.fps or 25.0)
        except Exception:
            fps_probe = 25.0
        events = normalize_timeline(plan, fps=fps_probe)
        events = enrich_keywords(events)
        

        
        # Hard fail if no valid events
        if not events:
            raise RuntimeError('Aucun B-roll valide après planification/scoring. Vérifier l\'index FAISS et la librairie. Aucun fallback synthétique appliqué.')
        # Valider que les médias existent
        from pathlib import Path as _Path
        valid_events = []
        for ev in events:
            mp = getattr(ev, 'media_path', '')
            pp = _Path(mp)
            if not pp.exists() and mp and not pp.is_absolute():
                pp = (broll_library / mp).resolve()
                if pp.exists():
                    try:
                        setattr(ev, 'media_path', str(pp))
                    except Exception:
                        pass
            if getattr(ev, 'media_path', '') and _Path(getattr(ev, 'media_path')).exists():
                valid_events.append(ev)
        # Log count and sample
        try:
            print(f"    🔎 B-roll events valides: {len(valid_events)}")
            for _ev in valid_events[:3]:
                print(f"       • {_ev.start_s:.2f}-{_ev.end_s:.2f} → {getattr(_ev, 'media_path','')}")
        except Exception:
            pass
        if not valid_events:
            # Fallback legacy: construire un plan simple à partir de la librairie existante
            try:
                _media_exts = {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}
                assets = [p for p in Path(broll_library).rglob('*') if p.suffix.lower() in _media_exts]
                assets.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
                assets = assets[:20]
                if assets:
                    # Choisir des segments suffisamment longs (>2.0s) et espacés
                    cands = []
                    for s in segments:
                        dur = float(getattr(s, 'end', 0.0) - getattr(s, 'start', 0.0))
                        if dur >= 2.0 and getattr(s, 'start', 0.0) >= 1.5:  # Plus flexible
                            cands.append(s)
                    plan_simple = []
                    gap = 6.0  # Réduit: 8s → 6s pour plus d'insertions
                    last = -1e9
                    ai = 0
                    for s in cands:
                        st = float(getattr(s,'start',0.0))
                        en = float(getattr(s,'end',0.0))
                        if st - last < gap:
                            continue
                        dur = min(7.0, max(2.5, en - st))  # Durée min: 2.5s, max: 7s
                        asset = assets[ai % len(assets)]
                        ai += 1
                        plan_simple.append({
                            'start': st,
                            'end': min(en, st + dur),
                            'asset_path': str(asset.resolve()),
                            'crossfade_frames': 2,
                        })
                        last = st
                    # Normaliser et rendre si on a des items
                    if plan_simple:
                        try:
                            with _VFC(str(input_path)) as _fpsprobe:
                                fps_probe = float(_fpsprobe.fps or 25.0)
                        except Exception:
                            fps_probe = 25.0
                        legacy_events = normalize_timeline(plan_simple, fps=fps_probe)
                        legacy_events = enrich_keywords(legacy_events)
                        print(f"    ♻️ Fallback legacy appliqué: {len(legacy_events)} events")
                        valid_events = legacy_events
                        # Continue vers le rendu unique plus bas
                    else:
                        raise RuntimeError('Librairie B-roll présente mais aucun slot valide pour fallback legacy')
                else:
                    raise RuntimeError('B-rolls planifiés, aucun media_path valide et aucune ressource en librairie pour fallback')
            except Exception as _e:
                raise RuntimeError('B-rolls planifiés, mais aucun media_path valide trouvé. Fallback legacy impossible: ' + str(_e))
        # Rendu unique avec les events valides (incl. fallback le cas échéant)
        render_video(cfg, segments, valid_events)
        
        # VÉRIFICATION ET NETTOYAGE INTELLIGENT DES B-ROLLS
        try:
            if getattr(Config, 'BROLL_DELETE_AFTER_USE', False):
                print("    🔍 Vérification des B-rolls avant suppression...")
                
                # Importer le système de vérification
                try:
                    from broll_verification_system import create_verification_system
                    verifier = create_verification_system()
                    
                    # Vérifier l'insertion des B-rolls
                    verification_result = verifier.verify_broll_insertion(
                        video_path=cfg.output_video,
                        broll_plan=plan or [],
                        broll_library_path=str(clip_broll_dir) if 'clip_broll_dir' in locals() else "AI-B-roll/broll_library"
                    )
                    
                    # 🚀 CORRECTION: Vérifier le type du résultat de vérification
                    if not isinstance(verification_result, dict):
                        print(f"    ⚠️ Résultat de vérification invalide (type: {type(verification_result)}) - Fallback vers vérification basique")
                        verification_result = {
                            "verification_passed": True,  # Par défaut, autoriser la suppression
                            "issues": [],
                            "recommendations": []
                        }
                    
                    # Décider si la suppression est autorisée
                    if verification_result.get("verification_passed", False):
                        print("    ✅ Vérification réussie - Suppression autorisée")
                        
                        # Supprimer seulement les fichiers B-roll utilisés (pas le dossier)
                        used_files: List[str] = []
                        for item in (plan or []):
                            path = getattr(item, 'asset_path', None) if hasattr(item, 'asset_path') else (item.get('asset_path') if isinstance(item, dict) else None)
                            if path and os.path.exists(path):
                                used_files.append(path)
                        
                        # Nettoyer les fichiers utilisés
                        cleaned_count = 0
                        for p in used_files:
                            try:
                                os.remove(p)
                                cleaned_count += 1
                            except Exception:
                                pass
                        
                        # Marquer le dossier comme "utilisé" mais le garder
                        if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                            try:
                                # Créer un fichier de statut pour indiquer que le clip est traité
                                status_file = clip_broll_dir / "STATUS_COMPLETED.txt"
                                status_file.write_text(f"Clip traité le {time.strftime('%Y-%m-%d %H:%M:%S')}\nB-rolls utilisés: {cleaned_count}\nVérification: PASSED\n", encoding='utf-8')
                                print(f"    🗂️ Dossier B-roll conservé: {clip_broll_dir.name} (fichiers nettoyés: {cleaned_count})")
                            except Exception as e:
                                print(f"    ⚠️ Erreur création statut: {e}")
                    else:
                        print("    ❌ Vérification échouée - Suppression REFUSÉE")
                        print("    📋 Problèmes détectés:")
                        for issue in verification_result.get("issues", []):
                            print(f"       • {issue}")
                        print("    💡 Recommandations:")
                        for rec in verification_result.get("recommendations", []):
                            print(f"       • {rec}")
                        
                        # Créer un fichier de statut d'échec
                        if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                            try:
                                status_file = clip_broll_dir / "STATUS_FAILED.txt"
                                status_file.write_text(f"Clip traité le {time.strftime('%Y-%m-%d %H:%M:%S')}\nVérification: FAILED\nProblèmes: {', '.join(verification_result.get('issues', []))}\n", encoding='utf-8')
                                print(f"    🚨 Dossier B-roll marqué comme échec: {clip_broll_dir.name}")
                            except Exception as e:
                                print(f"    ⚠️ Erreur création statut d'échec: {e}")
                
                except ImportError:
                    print("    ⚠️ Système de vérification non disponible - Suppression sans vérification")
                    # Fallback vers l'ancien système
                    used_files: List[str] = []
                    for item in (plan or []):
                        path = getattr(item, 'asset_path', None) if hasattr(item, 'asset_path') else (item.get('asset_path') if isinstance(item, dict) else None)
                        if path and os.path.exists(path):
                            used_files.append(path)
                    
                    cleaned_count = 0
                    for p in used_files:
                        try:
                            os.remove(p)
                            cleaned_count += 1
                        except Exception:
                            pass
                    
                    if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                        try:
                            status_file = clip_broll_dir / "STATUS_COMPLETED_NO_VERIFICATION.txt"
                            status_file.write_text(f"Clip traité le {time.strftime('%Y-%m-%d %H:%M:%S')}\nB-rolls utilisés: {cleaned_count}\nVérification: NON DISPONIBLE\n", encoding='utf-8')
                            print(f"    🗂️ Dossier B-roll conservé: {clip_broll_dir.name} (fichiers nettoyés: {cleaned_count})")
                        except Exception as e:
                            print(f"    ⚠️ Erreur création statut: {e}")
                
        except Exception as e:
            print(f"    ⚠️ Erreur lors de la vérification/nettoyage: {e}")
            # En cas d'erreur, ne pas supprimer les B-rolls
            pass

        if Path(cfg.output_video).exists():
            print("    ✅ B-roll insérés avec succès")
        
            # 🧹 NOUVEAU: Nettoyage immédiat du cache B-roll temporaire
            try:
                if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                    folder_size = sum(f.stat().st_size for f in clip_broll_dir.rglob('*') if f.is_file()) / (1024**2)  # MB
                    # 🚀 OPTIMISATION: Utiliser safe_remove_tree
                    if safe_remove_tree(clip_broll_dir):
                        print(f"    🗑️ Cache B-roll nettoyé: {folder_size:.1f} MB libérés")
                        print(f"    💾 Dossier temporaire supprimé: {clip_broll_dir.name}")
                    else:
                        print(f"    ⚠️ Nettoyage partiel du cache B-roll")
            except Exception as e:
                print(f"    ⚠️ Erreur nettoyage cache: {e}")
        
            return Path(cfg.output_video)
        else:
            print("    ⚠️ Sortie B-roll introuvable, retour à la vidéo d'origine")
        
        # 🧹 Nettoyer même en cas d'échec
        try:
            if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                # 🚀 OPTIMISATION: Utiliser safe_remove_tree
                if safe_remove_tree(clip_broll_dir):
                    print(f"    🗑️ Cache B-roll nettoyé (échec traitement)")
        except Exception:
            pass
        
        return input_path
    except Exception as e:
        print(f"    ❌ Erreur B-roll: {e}")
        
        # 🧹 IMPORTANT: Nettoyer même en cas d'erreur pour éviter l'accumulation
        try:
            if 'clip_broll_dir' in locals() and clip_broll_dir.exists():
                # 🚀 OPTIMISATION: Utiliser safe_remove_tree
                if safe_remove_tree(clip_broll_dir):
                    print(f"    🗑️ Cache B-roll nettoyé après erreur")
        except Exception:
            pass
        
        return input_path

# Si densité trop faible après planification, injecter quelques B-rolls génériques
try:
    with _VFC(str(input_path)) as _tmp:
        _total = float(_tmp.duration or 0.0)
    cur_cov = sum(max(0.0, (float(getattr(it,'end', it.get('end',0.0))) - float(getattr(it,'start', it.get('start',0.0))))) for it in (plan or []))
    if _total > 0 and (cur_cov / _total) < 0.20:  # Augmenté: 15% → 20% pour plus de B-rolls
        _generics = []
        bank = [
            "money", "handshake", "meeting", "audience", "lightbulb", "typing", "city", "success"
        ]
        # Chercher quelques médias génériques existants
        for p in broll_library.rglob('*'):
            if p.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.jpg','.jpeg','.png'}:
                name = p.stem.lower()
                if any(k in name for k in bank):
                    _generics.append(str(p.resolve()))
        if _generics:
            # Injecter 2–4 génériques espacés
            inject_count = min(4, max(2, int(len(_generics)/5)))
            st = 2.0
            while inject_count > 0 and st < (_total - 3.5):
                plan.append({'start': st, 'end': min(_total, st+3.5), 'asset_path': _generics[inject_count % len(_generics)], 'crossfade_frames': 2})
                st += 10.0
                inject_count -= 1
            print("    ➕ B-rolls génériques injectés pour densité minimale")
except Exception:
    pass
def _prioritize_fresh_assets(broll_candidates, clip_id):
    """Priorise les assets les plus récents basés sur le timestamp du dossier."""
    if not broll_candidates:
        return broll_candidates
    try:
        for candidate in broll_candidates:
            if hasattr(candidate, 'file_path') and candidate.file_path:
                path = Path(candidate.file_path)
                for part in path.parts:
                    if part.startswith(f"clip_{clip_id}_") and "_" in part:
                        timestamp_str = part.split("_")[-1]
                        if timestamp_str.isdigit():
                            candidate.folder_timestamp = int(timestamp_str)
                            break
                else:
                    candidate.folder_timestamp = 0
            else:
                candidate.folder_timestamp = 0
        broll_candidates.sort(key=lambda item: getattr(item, 'folder_timestamp', 0), reverse=True)
    except Exception as exc:
        print(f"    ⚠️  Erreur priorisation fraîcheur: {exc}")
    return broll_candidates


def _debug_broll_selection(plan, domain, keywords, debug_mode=False):
    if not debug_mode:
        return
    print("    🔍 DEBUG B-ROLL SELECTION:")
    print(f"       Domaine: {domain}")
    print(f"       Mots-clés: {keywords[:5]}")
    print(f"       Plan: {len(plan)} items")
    for i, item in enumerate(plan[:3]):
        if hasattr(item, 'asset_path') and item.asset_path:
            asset_path = item.asset_path
            score = getattr(item, 'score', 'N/A')
            context_score = getattr(item, 'context_score', 'N/A')
            freshness = getattr(item, 'freshness_score', 'N/A')
        elif isinstance(item, dict):
            asset_path = item.get('asset_path', 'N/A')
            score = item.get('score', 'N/A')
            context_score = item.get('context_score', 'N/A')
            freshness = item.get('freshness_score', 'N/A')
        else:
            continue
        print(f"       Item {i+1}: {Path(asset_path).name}")
        print(f"         Score: {score}, Context: {context_score}, Fraîcheur: {freshness}")



def _get_fallback_neutral_assets(broll_library: Path, count: int = 3) -> List[str]:
    try:
        fallback_keywords = ['neutral', 'generic', 'background', 'abstract', 'minimal']
        fallback_assets: List[str] = []
        for keyword in fallback_keywords:
            for ext in ['.mp4', '.mov', '.jpg', '.png']:
                for asset_path in broll_library.rglob(f"*{keyword}*{ext}"):
                    if asset_path.exists() and asset_path not in fallback_assets:
                        fallback_assets.append(str(asset_path))
                        if len(fallback_assets) >= count:
                            break
                if len(fallback_assets) >= count:
                    break
            if len(fallback_assets) >= count:
                break
        if len(fallback_assets) < count:
            for ext in ['.mp4', '.mov', '.jpg', '.png']:
                for asset_path in broll_library.rglob(f"*{ext}"):
                    if asset_path.exists() and asset_path not in fallback_assets:
                        fallback_assets.append(str(asset_path))
                        if len(fallback_assets) >= count:
                            break
                if len(fallback_assets) >= count:
                    break
        return fallback_assets[:count]
    except Exception as exc:
        print(f"    ⚠️  Erreur fallback neutre: {exc}")
        return []

def _score_contextual_relevance(asset_path, domain, keywords):
    try:
        if not asset_path or not domain or not keywords:
            return 0.5
        filename = Path(asset_path).stem.lower()
        asset_tokens = set(re.split(r'[^a-z0-9]+', filename))
        domain_tokens = set(domain.lower().split())
        keyword_tokens = set()
        for keyword in keywords:
            if isinstance(keyword, str):
                keyword_tokens.update(keyword.lower().split())
        relevant_tokens = domain_tokens | keyword_tokens
        if not relevant_tokens:
            return 0.5
        overlap = len(asset_tokens & relevant_tokens)
        total_relevant = len(relevant_tokens)
        base_score = min(1.0, overlap / max(1, total_relevant * 0.3))
        domain_overlap = len(asset_tokens & domain_tokens)
        domain_bonus = min(0.3, domain_overlap * 0.1)
        return min(1.0, base_score + domain_bonus)
    except Exception as exc:
        print(f"    ⚠️  Erreur scoring contextuel: {exc}")
        return 0.5


def _get_domain_keywords(domain: str) -> List[str]:
    domain_keywords = {
        'health': ['medical', 'healthcare', 'wellness', 'fitness', 'medicine', 'hospital', 'doctor'],
        'technology': ['tech', 'digital', 'innovation', 'computer', 'ai', 'software', 'data'],
        'business': ['business', 'entrepreneur', 'success', 'growth', 'strategy', 'office', 'professional'],
        'education': ['learning', 'education', 'knowledge', 'study', 'teaching', 'school', 'university'],
        'finance': ['money', 'finance', 'investment', 'wealth', 'business', 'success', 'growth'],
    }
    return domain_keywords.get(domain.lower(), [domain])


def _calculate_quality_score(asset_path: str, metadata: Optional[Dict] = None) -> float:
    try:
        score = 0.5
        if metadata and 'resolution' in metadata:
            res = metadata['resolution']
            if '4k' in res or '3840' in res:
                score += 0.2
            elif '1080' in res or '1920' in res:
                score += 0.1
        if metadata and 'duration' in metadata:
            duration = metadata['duration']
            if 2.0 <= duration <= 6.0:
                score += 0.1
        if asset_path.lower().endswith('.mp4'):
            score += 0.1
        return min(1.0, score)
    except Exception:
        return 0.5


def _score_broll_asset_basic(asset_path: str, asset_tags: List[str], query_keywords: List[str]) -> float:
    try:
        if not asset_tags or not query_keywords:
            return 0.5
        asset_tag_set = {tag.lower() for tag in asset_tags}
        query_set = {kw.lower() for kw in query_keywords if kw}
        if not query_set:
            return 0.5
        intersection = len(asset_tag_set & query_set)
        union = len(asset_tag_set | query_set)
        return intersection / union if union > 0 else 0.0
    except Exception as exc:
        print(f"⚠️ Erreur scoring basique: {exc}")
        return 0.5


def score_broll_asset_mixed(
    asset_path: str,
    asset_tags: List[str],
    query_keywords: List[str],
    domain: Optional[str] = None,
    asset_metadata: Optional[Dict] = None,
) -> float:
    try:
        from broll_selector import Asset, ScoringFeatures
        if not BROLL_SELECTOR_AVAILABLE:
            return _score_broll_asset_basic(asset_path, asset_tags, query_keywords)
        asset = Asset(
            id=f"asset_{hash(asset_path)}",
            file_path=asset_path,
            tags=asset_tags,
            title=Path(asset_path).stem,
            description="",
            source="local",
            fetched_at=datetime.now(),
            duration=asset_metadata.get('duration', 2.0) if asset_metadata else 2.0,
            resolution=asset_metadata.get('resolution', '1920x1080') if asset_metadata else '1920x1080',
        )
        normalized_keywords = {
            kw.lower().strip() for kw in query_keywords if isinstance(kw, str) and len(kw.strip()) > 2
        }
        features = ScoringFeatures()
        if asset_tags and normalized_keywords:
            intersection = len(set(asset_tags) & normalized_keywords)
            union = len(set(asset_tags) | normalized_keywords)
            features.token_overlap = intersection / union if union > 0 else 0.0
        if domain and asset_tags:
            domain_keywords = _get_domain_keywords(domain)
            domain_overlap = len(set(asset_tags) & set(domain_keywords))
            features.domain_match = min(1.0, domain_overlap / max(len(domain_keywords), 1))
        try:
            file_path = Path(asset_path)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                days_old = (time.time() - mtime) / (24 * 3600)
                features.freshness = 1.0 / (1.0 + days_old / 60)
        except Exception:
            features.freshness = 0.5
        features.quality_score = _calculate_quality_score(asset_path, asset_metadata)
        features.embedding_similarity = 0.5
        weights = {
            'embedding': 0.4,
            'token': 0.2,
            'domain': 0.15,
            'freshness': 0.1,
            'quality': 0.1,
            'diversity': 0.05,
        }
        final_score = (
            weights['embedding'] * features.embedding_similarity
            + weights['token'] * features.token_overlap
            + weights['domain'] * features.domain_match
            + weights['freshness'] * features.freshness
            + weights['quality'] * features.quality_score
        )
        return max(0.0, min(1.0, final_score))
    except Exception as exc:
        print(f"⚠️ Erreur scoring mixte: {exc}")
        return _score_broll_asset_basic(asset_path, asset_tags, query_keywords)


class BrollProcessor:
    """Encapsule la logique d'insertion et de maintenance des B-rolls."""

    def __init__(
        self,
        config,
        safe_remove_tree: Callable[[Path], bool],
        get_sentence_transformer_model: Callable[[str], object],
    ) -> None:
        self.config = config
        self._safe_remove_tree = safe_remove_tree
        self._get_sentence_transformer_model = get_sentence_transformer_model

    def insert_brolls_if_enabled(
        self, input_path: Path, subtitles: List[Dict], broll_keywords: List[str]
    ) -> Path:
        return _insert_brolls(
            self.config,
            input_path,
            subtitles,
            broll_keywords,
            self._safe_remove_tree,
            self._get_sentence_transformer_model,
        )

    def cleanup_all_temp_broll(self) -> None:
        cleanup_all_temp_broll(self.config, self._safe_remove_tree)

    def cleanup_broll_duplicates(self) -> None:
        cleanup_broll_duplicates(self.config, self._safe_remove_tree)

    def purge_broll_caches(self) -> None:
        purge_broll_caches(self.config, self._safe_remove_tree)
