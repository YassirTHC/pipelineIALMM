import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import gc
import os
import json
import subprocess
import random
import numpy as np
import shutil
import requests
import cv2
from datetime import datetime
import whisper

# 🚀 NOUVEAU: Configuration des logs temps réel + suppression warnings non-critiques
import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")
warnings.filterwarnings("ignore", message="Warning: in file.*bytes wanted but.*bytes read")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 🚀 NOUVEAU: Fonction print temps réel
def print_realtime(message):
    """Print avec flush immédiat pour logs temps réel"""
    print(message, flush=True)
    logger.info(message)

from pipeline.reframe import ReframeProcessor, configure_imagemagick
from pipeline.subtitles import SubtitleProcessor, write_srt, write_vtt
from pipeline.broll import BrollProcessor

configure_imagemagick()

# 🚀 NOUVEAU: Cache global pour éviter le rechargement des modèles
_MODEL_CACHE = {}

def get_sentence_transformer_model(model_name: str):
    """Récupère un modèle SentenceTransformer depuis le cache ou le charge"""
    # 🚀 OPTIMISATION: Normaliser le nom du modèle pour éviter les doublons
    normalized_name = model_name.replace('sentence-transformers/', '')
    
    if normalized_name not in _MODEL_CACHE:
        print(f"    🔄 Chargement initial du modèle: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[normalized_name] = SentenceTransformer(model_name)
            print(f"    ✅ Modèle {model_name} chargé et mis en cache")
        except Exception as e:
            print(f"    ❌ Erreur chargement modèle {model_name}: {e}")
            return None
    else:
        print(f"    ♻️ Modèle {model_name} récupéré du cache")
    
    return _MODEL_CACHE[normalized_name]

def safe_remove_tree(directory: Path, max_retries: int = 3, delay: float = 1.0) -> bool:
    """
    Supprime un dossier de façon sécurisée avec retry et gestion des handles Windows
    
    Args:
        directory: Dossier à supprimer
        max_retries: Nombre maximum de tentatives
        delay: Délai entre les tentatives (secondes)
    
    Returns:
        True si la suppression a réussi, False sinon
    """
    if not directory.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            # Forcer la libération des handles
            gc.collect()
            
            # Tentative de suppression récursive
            shutil.rmtree(directory, ignore_errors=False)
            
            # Vérifier que c'est vraiment supprimé
            if not directory.exists():
                return True
                
        except PermissionError as e:
            if "WinError 32" in str(e) or "being used by another process" in str(e):
                print(f"    ⚠️ Tentative {attempt + 1}/{max_retries}: Fichier en cours d'utilisation, retry dans {delay}s...")
                time.sleep(delay)
                delay *= 1.5  # Backoff exponentiel
                continue
            else:
                print(f"    ❌ Erreur de permission: {e}")
                break
        except Exception as e:
            print(f"    ❌ Erreur inattendue lors de la suppression: {e}")
            break
    
    # Si on arrive ici, toutes les tentatives ont échoué
    try:
        # Tentative finale avec ignore_errors=True
        shutil.rmtree(directory, ignore_errors=True)
        if not directory.exists():
            print(f"    ✅ Suppression réussie avec ignore_errors")
            return True
        else:
            print(f"    ⚠️ Dossier partiellement supprimé, résidu: {directory}")
            return False
    except Exception as e:
        print(f"    ❌ Échec final de suppression: {e}")
        return False

# Gestion optionnelle de Mediapipe avec fallback
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ Mediapipe disponible - Utilisation des fonctionnalités IA avancées")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("⚠️ Mediapipe non disponible - Utilisation du fallback OpenCV (fonctionnalités réduites)")

# 🚀 NOUVEAU: Import du sélecteur B-roll générique
try:
    from broll_selector import BrollSelector, Asset, ScoringFeatures, BrollCandidate
    BROLL_SELECTOR_AVAILABLE = True
    print("✅ Sélecteur B-roll générique disponible - Scoring mixte activé")
except ImportError as e:
    BROLL_SELECTOR_AVAILABLE = False
    print(f"⚠️ Sélecteur B-roll générique non disponible: {e}")
    print("   🔄 Utilisation du système de scoring existant")

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from tqdm import tqdm  # NEW: console progress
import re # NEW: for caption/hashtag generation
from hormozi_subtitles import add_hormozi_subtitles


def _read_ui_settings() -> Dict:
    """Read optional UI settings from config/ui_settings.json."""
    try:
        cfg_path = Path('config/ui_settings.json')
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _to_bool(v, default=False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1","true","yes","on"}


# Load optional UI overrides once
_UI_SETTINGS = _read_ui_settings()

# Configuration automatique d'ImageMagick pour MoviePy
class Config:
    """Configuration centralisée du pipeline"""
    CLIPS_FOLDER = Path("./clips")
    OUTPUT_FOLDER = Path("./output") 
    TEMP_FOLDER = Path("./temp")
    
    # Résolution cible pour les réseaux sociaux
    TARGET_WIDTH = 720
    TARGET_HEIGHT = 1280  # Format 9:16
    
    # Paramètres Whisper
    WHISPER_MODEL = "tiny"  # ou "small", "medium", "large"
    
    # Paramètres sous-titres
    SUBTITLE_FONT_SIZE = 85
    SUBTITLE_COLOR = 'yellow'
    SUBTITLE_STROKE_COLOR = 'black'
    SUBTITLE_STROKE_WIDTH = 3
    # Biais global (en secondes) pour corriger un léger décalage systématique
    # 0.0 par défaut pour éviter tout décalage si non nécessaire
    SUBTITLE_TIMING_BIAS_S = 0.0

    # Activation B-roll: UI > ENV > défaut(off)
    # Si fetchers cochés, activer automatiquement l'insertion B-roll, sauf si explicitement désactivé côté UI
    _UI_ENABLE_BROLL = _UI_SETTINGS.get('enable_broll') if 'enable_broll' in _UI_SETTINGS else None
    _ENV_ENABLE_BROLL = os.getenv('ENABLE_BROLL') or os.getenv('AI_BROLL_ENABLED')
    _AUTO_ENABLE = _to_bool(_UI_SETTINGS.get('broll_fetch_enable'), default=True) if 'broll_fetch_enable' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ENABLE') or os.getenv('AI_BROLL_ENABLE_FETCHER'), default=True)
    ENABLE_BROLL = (
        _to_bool(_UI_ENABLE_BROLL, default=False) if _UI_ENABLE_BROLL is not None
        else (_to_bool(_ENV_ENABLE_BROLL, default=False) or _AUTO_ENABLE)
    )

    # === Options fetcher B-roll (stock) ===
    # Active le fetch automatique: UI > ENV > défaut(on)
    BROLL_FETCH_ENABLE = _to_bool(_UI_SETTINGS.get('broll_fetch_enable'), default=True) if 'broll_fetch_enable' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ENABLE') or os.getenv('AI_BROLL_ENABLE_FETCHER'), default=True)
    # Fournisseur: UI > ENV > défaut pexels
    BROLL_FETCH_PROVIDER = (_UI_SETTINGS.get('broll_fetch_provider') or os.getenv('AI_BROLL_FETCH_PROVIDER') or 'pexels')
    # Clés API
    PEXELS_API_KEY = _UI_SETTINGS.get('PEXELS_API_KEY') or os.getenv('PEXELS_API_KEY')
    PIXABAY_API_KEY = _UI_SETTINGS.get('PIXABAY_API_KEY') or os.getenv('PIXABAY_API_KEY')
    # Contrôles de fetch
    BROLL_FETCH_MAX_PER_KEYWORD = int(_UI_SETTINGS.get('broll_fetch_max_per_keyword') or os.getenv('BROLL_FETCH_MAX_PER_KEYWORD') or 25)  # CORRIGÉ: 12 → 25
    BROLL_FETCH_ALLOW_VIDEOS = _to_bool(_UI_SETTINGS.get('broll_fetch_allow_videos'), default=True) if 'broll_fetch_allow_videos' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ALLOW_VIDEOS'), default=True)
    BROLL_FETCH_ALLOW_IMAGES = _to_bool(_UI_SETTINGS.get('broll_fetch_allow_images'), default=False) if 'broll_fetch_allow_images' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_FETCH_ALLOW_IMAGES'), default=False)
    # Élargir le pool par défaut: activer les images si non précisé
    if 'broll_fetch_allow_images' not in _UI_SETTINGS and os.getenv('BROLL_FETCH_ALLOW_IMAGES') is None:
        BROLL_FETCH_ALLOW_IMAGES = True
    # Embeddings pour matching sémantique
    BROLL_USE_EMBEDDINGS = _to_bool(_UI_SETTINGS.get('broll_use_embeddings'), default=True) if 'broll_use_embeddings' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_USE_EMBEDDINGS'), default=True)
    BROLL_EMBEDDING_MODEL = (_UI_SETTINGS.get('broll_embedding_model') or os.getenv('BROLL_EMBEDDING_MODEL') or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # Config contextuelle
    CONTEXTUAL_CONFIG_PATH = Path(_UI_SETTINGS.get('contextual_broll_yml') or os.getenv('CONTEXTUAL_BROLL_YML') or 'config/contextual_broll.yml')

    # Sortie et nettoyage
    USE_HARDLINKS = _to_bool(_UI_SETTINGS.get('use_hardlinks'), default=True) if 'use_hardlinks' in _UI_SETTINGS else _to_bool(os.getenv('USE_HARDLINKS'), default=True)
    BROLL_DELETE_AFTER_USE = _to_bool(_UI_SETTINGS.get('broll_delete_after_use'), default=True) if 'broll_delete_after_use' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_DELETE_AFTER_USE') or os.getenv('AI_BROLL_PURGE_AFTER_USE'), default=True)
    # 🚀 NOUVEAU: Forcer le nettoyage après chaque vidéo pour économiser l'espace
    BROLL_CLEANUP_PER_VIDEO = True  # Toujours activé pour éviter l'accumulation
    BROLL_PURGE_AFTER_RUN = _to_bool(_UI_SETTINGS.get('broll_purge_after_run'), default=True) if 'broll_purge_after_run' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_PURGE_AFTER_RUN') or os.getenv('AI_BROLL_PURGE_AFTER_RUN'), default=True)
    # Brand kit
    BRAND_KIT_ID = _UI_SETTINGS.get('brand_kit_id') or os.getenv('BRAND_KIT_ID') or 'default'
    # Experimental FX (wipes/zoom/LUT etc.)
    ENABLE_EXPERIMENTAL_FX = _to_bool(_UI_SETTINGS.get('enable_experimental_fx'), default=False) if 'enable_experimental_fx' in _UI_SETTINGS else _to_bool(os.getenv('ENABLE_EXPERIMENTAL_FX'), default=False)

    # 🚀 NOUVEAU: Configuration du sélecteur B-roll générique
    BROLL_SELECTOR_CONFIG_PATH = Path(_UI_SETTINGS.get('broll_selector_config') or os.getenv('BROLL_SELECTOR_CONFIG') or 'config/broll_selector_config.yaml')
    BROLL_SELECTOR_ENABLED = _to_bool(_UI_SETTINGS.get('broll_selector_enabled'), default=True) if 'broll_selector_enabled' in _UI_SETTINGS else _to_bool(os.getenv('BROLL_SELECTOR_ENABLED') or os.getenv('AI_BROLL_SELECTOR_ENABLED'), default=True)

# 🚀 SUPPRIMÉ: Fonction _detect_local_llm obsolète
# Remplacée par le système LLM industriel qui gère automatiquement la détection

# 🚀 SUPPRIMÉ: Ancien système LLM obsolète remplacé par le système industriel
# Cette fonction utilisait l'ancien prompt complexe et causait des timeouts
# Maintenant remplacée par le système LLM industriel dans generate_caption_and_hashtags
# 🚀 SUPPRIMÉ: Reste de l'ancien système LLM obsolète
# Toute cette logique complexe est maintenant remplacée par le système industriel

# === IA: Analyse mots-clés et prompts visuels pour guider le B-roll ===

class VideoProcessor:
    """Classe principale pour traiter les vidéos"""
    
    def __init__(self):
        self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
        self._setup_directories()
        self.reframer = ReframeProcessor(Config, MEDIAPIPE_AVAILABLE, mp, logger, print_realtime)
        self.subtitle_processor = SubtitleProcessor(Config, self.whisper_model, logger, print_realtime)
        self.broll_processor = BrollProcessor(Config, safe_remove_tree, get_sentence_transformer_model)

    def reframe_to_vertical(self, clip_path: Path) -> Path:
        return self.reframer.reframe_to_vertical(clip_path)

    def transcribe_segments(self, video_path: Path) -> List[Dict]:
        return self.subtitle_processor.transcribe_segments(video_path)

    def generate_caption_and_hashtags(self, subtitles: List[Dict]):
        return self.subtitle_processor.generate_caption_and_hashtags(subtitles)

    def insert_brolls_if_enabled(self, input_path: Path, subtitles: List[Dict], broll_keywords: List[str]) -> Path:
        return self.broll_processor.insert_brolls_if_enabled(input_path, subtitles, broll_keywords)

    
    def _setup_directories(self):
        """Crée les dossiers nécessaires"""
        for folder in [Config.CLIPS_FOLDER, Config.OUTPUT_FOLDER, Config.TEMP_FOLDER]:
            folder.mkdir(exist_ok=True)
    
    def _generate_unique_output_dir(self, clip_stem: str) -> Path:
        """Crée un dossier unique pour ce clip sous output/clips/<stem>[-NNN]"""
        root = Config.OUTPUT_FOLDER / 'clips'
        root.mkdir(parents=True, exist_ok=True)
        base = root / clip_stem
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)
            return base
        # Trouver suffixe -001, -002, ...
        for i in range(1, 1000):
            candidate = root / f"{clip_stem}-{i:03d}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
        # Fallback timestamp
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        cand = root / f"{clip_stem}-{ts}"
        cand.mkdir(parents=True, exist_ok=True)
        return cand
    
    def _safe_copy(self, src: Path, dst: Path) -> None:
        try:
            if src and Path(src).exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
        except Exception:
            pass

    def _hardlink_or_copy(self, src: Path, dst: Path) -> None:
        """Crée un hardlink si possible, sinon copie le fichier."""
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if getattr(Config, 'USE_HARDLINKS', True):
                os.link(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
        except Exception:
            try:
                shutil.copy2(str(src), str(dst))
            except Exception:
                pass
 
    def _unique_path(self, directory: Path, base_name: str, extension: str) -> Path:
        """Retourne un chemin unique dans directory en ajoutant -NNN si collision."""
        directory.mkdir(parents=True, exist_ok=True)
        candidate = directory / f"{base_name}{extension}"
        if not candidate.exists():
            return candidate
        for i in range(1, 1000):
            alt = directory / f"{base_name}-{i:03d}{extension}"
            if not alt.exists():
                return alt
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        return directory / f"{base_name}-{ts}{extension}"
    
    def _cleanup_files(self, paths: List[Path]) -> None:
        for p in paths:
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass
 
    def process_all_clips(self, input_video_path: str):
        """Pipeline principal de traitement"""
        logger.info("🚀 Début du pipeline de traitement")
        print("🎬 Démarrage du pipeline de traitement...")
        
        # Étape 1: Découpage (votre IA existante)
        
        # Étape 2: Traitement de chaque clip
        clip_files = list(Config.CLIPS_FOLDER.glob("*.mp4"))
        total_clips = len(clip_files)
        
        print(f"📁 {total_clips} clips trouvés dans le dossier clips/")
        
        for i, clip_path in enumerate(clip_files):
            print(f"\n🎬 [{i+1}/{total_clips}] Traitement de: {clip_path.name}")
            logger.info(f"🎬 Traitement du clip {i+1}/{total_clips}: {clip_path.name}")
            
            # Skip si déjà traité
            stem = Path(clip_path).stem
            final_dir = Config.OUTPUT_FOLDER / 'final'
            processed_already = False
            if final_dir.exists():
                matches = list(final_dir.glob(f"final_{stem}*.mp4"))
                processed_already = len(matches) > 0
            if processed_already:
                print(f"⏩ Clip déjà traité, ignoré : {clip_path.name}")
                logger.info(f"⏩ Clip déjà traité, ignoré : {clip_path.name}")
                continue

            # Verrou concurrentiel par clip
            locks_dir = Config.OUTPUT_FOLDER / 'locks'
            locks_dir.mkdir(parents=True, exist_ok=True)
            lock_file = locks_dir / f"{stem}.lock"
            if lock_file.exists():
                print(f"⏭️ Verrou détecté, saut du clip: {clip_path.name}")
                continue
            try:
                lock_file.write_text("locked", encoding='utf-8')
                self.process_single_clip(clip_path)
                print(f"✅ Clip {clip_path.name} traité avec succès")
                logger.info(f"✅ Clip {clip_path.name} traité avec succès")
            except Exception as e:
                print(f"❌ Erreur lors du traitement de {clip_path.name}: {e}")
                logger.error(f"❌ Erreur lors du traitement de {clip_path.name}: {e}")
            finally:
                try:
                    if lock_file.exists():
                        lock_file.unlink()
                except Exception:
                    pass
        
        print(f"\n🎉 Pipeline terminé ! {total_clips} clips traités.")
        logger.info("🎉 Pipeline terminé avec succès")
        # 🧹 NETTOYAGE AUTOMATIQUE AGRESSIF pour éviter l'accumulation
        try:
            # 🚀 NOUVEAU: Nettoyage systématique après chaque session
            if getattr(Config, 'BROLL_CLEANUP_PER_VIDEO', True):
                print("🧹 Nettoyage automatique de tous les caches B-roll temporaires...")
                self.broll_processor.cleanup_all_temp_broll()
            
            # Nettoyage traditionnel si activé
            if getattr(Config, 'BROLL_PURGE_AFTER_RUN', False):
                self.broll_processor.purge_broll_caches()
            
            # Nettoyage des doublons résiduels
            self.broll_processor.cleanup_broll_duplicates()
        except Exception as e:
            print(f"⚠️ Erreur nettoyage automatique: {e}")
        # Agréger un rapport global même sans --json-report
        try:
            final_dir = (Config.OUTPUT_FOLDER / 'final')
            items = []
            if final_dir.exists():
                for jf in final_dir.glob('final_*.json'):
                    try:
                        items.append(json.loads(jf.read_text(encoding='utf-8')))
                    except Exception:
                        pass
            report_path = Config.OUTPUT_FOLDER / 'report.json'
            report_path.write_text(json.dumps({'clips': items}, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    def cut_viral_clips(self, input_video_path: str):
        """
        Interface pour votre IA de découpage existante
        Remplacez cette méthode par votre implémentation
        """
        logger.info("📼 Découpage des clips avec IA...")
        
        # Exemple basique - remplacez par votre IA
        video = VideoFileClip(input_video_path)
        duration = video.duration
        
        # Découpage adaptatif selon la durée
        if duration <= 30:
            # Vidéo courte : utiliser toute la vidéo
            segment_duration = duration
            segments = 1
        else:
            # Vidéo longue : découper en segments de 30 secondes
            segment_duration = 30
            segments = max(1, int(duration // segment_duration))
        
        for i in range(min(segments, 5)):  # Max 5 clips pour test
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            
            clip = video.subclip(start_time, end_time)
            output_path = Config.CLIPS_FOLDER / f"clip_{i+1:02d}.mp4"
            clip.write_videofile(str(output_path), verbose=False, logger=None)
        
        video.close()
        logger.info(f"✅ {segments} clips générés")
    
    def process_single_clip(self, clip_path: Path):
        """Traite un clip individuel (reframe -> transcription (pour B-roll) -> B-roll -> sous-titres)"""
        
        # Dossier de sortie dédié et unique
        per_clip_dir = self._generate_unique_output_dir(clip_path.stem)
        
        print(f"  📐 Étape 1/4: Reframe dynamique IA...")
        reframed_path = self.reframer.reframe_to_vertical(clip_path)
        # Déplacer artefact reframed dans le dossier du clip
        try:
            dst_reframed = per_clip_dir / 'reframed.mp4'
            if Path(reframed_path).exists():
                shutil.move(str(reframed_path), str(dst_reframed))
            reframed_path = dst_reframed
        except Exception:
            pass
        
        print(f"  🗣️ Étape 2/4: Transcription Whisper (guide B-roll)...")
        # Transcrire tôt pour guider la sélection B-roll (SRT disponible)
        subtitles = self.subtitle_processor.transcribe_segments(reframed_path)
        try:
            # Écrire un SRT à côté de la vidéo reframée
            srt_reframed = reframed_path.with_suffix('.srt')
            write_srt(subtitles, srt_reframed)
            # Sauvegarder transcription segments JSON
            seg_json = per_clip_dir / f"{clip_path.stem}_segments.json"
            with open(seg_json, 'w', encoding='utf-8') as f:
                json.dump(subtitles, f, ensure_ascii=False)
        except Exception:
            pass
        
        print(f"  🎞️ Étape 3/4: Insertion des B-rolls {'(activée)' if getattr(Config, 'ENABLE_BROLL', False) else '(désactivée)'}...")
        
        # 🚀 CORRECTION: Générer les mots-clés LLM AVANT l'insertion des B-rolls
        broll_keywords = []
        try:
            print("    🤖 Génération précoce des mots-clés LLM pour B-rolls...")
            title, description, hashtags, broll_keywords = self.subtitle_processor.generate_caption_and_hashtags(subtitles)
            print(f"    ✅ Mots-clés B-roll LLM générés: {len(broll_keywords)} termes")
            print(f"    🎯 Exemples: {', '.join(broll_keywords[:5])}")
        except Exception as e:
            print(f"    ⚠️ Erreur génération mots-clés LLM: {e}")
            broll_keywords = []
        
        # Maintenant insérer les B-rolls avec les mots-clés LLM disponibles
        with_broll_path = self.broll_processor.insert_brolls_if_enabled(reframed_path, subtitles, broll_keywords)
        
        # Copier artefact with_broll si différent
        try:
            if with_broll_path and with_broll_path != reframed_path:
                self._safe_copy(with_broll_path, per_clip_dir / 'with_broll.mp4')
        except Exception:
            pass
        
        print(f"  ✨ Étape 4/4: Ajout des sous-titres Hormozi 1...")
        # Générer meta (titre/hashtags) depuis transcription (déjà fait)
        try:
            # Réutiliser les données déjà générées
            if not broll_keywords:  # Fallback si pas encore généré
                title, description, hashtags, broll_keywords = self.subtitle_processor.generate_caption_and_hashtags(subtitles)
            
            print(f"  📝 Title: {title}")
            print(f"  📝 Description: {description}")
            print(f"  #️⃣ Hashtags: {' '.join(hashtags)}")
            meta_path = per_clip_dir / 'meta.txt'
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(
                    "Title: " + title + "\n\n" +
                    "Description: " + description + "\n\n" +
                    "Hashtags: " + ' '.join(hashtags) + "\n\n" +
                    "B-roll Keywords: " + ', '.join(broll_keywords) + "\n"
                )
            print(f"  📝 [MÉTADONNÉES] Fichier meta.txt sauvegardé: {meta_path}")
        except Exception as e:
            print(f"  ⚠️ [ERREUR MÉTADONNÉES] {e}")
            # Fallback: créer des métadonnées basiques
            try:
                meta_path = per_clip_dir / 'meta.txt'
                with open(meta_path, 'w', encoding='utf-8') as f:
                    f.write("Title: Vidéo générée automatiquement\n\nDescription: Contenu généré par pipeline vidéo\n\nHashtags: #video #auto\n\nB-roll Keywords: video, content\n")
                print(f"  📝 [FALLBACK] Métadonnées de base sauvegardées: {meta_path}")
            except Exception as e2:
                print(f"  ❌ [ERREUR FALLBACK] {e2}")
        
        # Appliquer style Hormozi sur la vidéo post B-roll
        subtitled_out_dir = per_clip_dir
        subtitled_out_dir.mkdir(parents=True, exist_ok=True)
        final_subtitled_path = subtitled_out_dir / 'final_subtitled.mp4'
        try:
            span_style_map = {
                # Business & Croissance
                "croissance": {"color": "#39FF14", "bold": True, "emoji": "📈"},
                "growth": {"color": "#39FF14", "bold": True, "emoji": "📈"},
                "opportunité": {"color": "#FFD700", "bold": True, "emoji": "��"},
                "opportunite": {"color": "#FFD700", "bold": True, "emoji": "🔑"},
                "innovation": {"color": "#00E5FF", "emoji": "⚡"},
                "idée": {"color": "#00E5FF", "emoji": "💡"},
                "idee": {"color": "#00E5FF", "emoji": "💡"},
                "stratégie": {"color": "#FF73FA", "emoji": "🧭"},
                "strategie": {"color": "#FF73FA", "emoji": "🧭"},
                "plan": {"color": "#FF73FA", "emoji": "🗺️"},
                # Argent & Finance
                "argent": {"color": "#FFD700", "bold": True, "emoji": "💰"},
                "money": {"color": "#FFD700", "bold": True, "emoji": "💰"},
                "cash": {"color": "#FFD700", "bold": True, "emoji": "💰"},
                "investissement": {"color": "#8AFF00", "bold": True, "emoji": "📊"},
                "investissements": {"color": "#8AFF00", "bold": True, "emoji": "📊"},
                "revenu": {"color": "#8AFF00", "emoji": "🏦"},
                "revenus": {"color": "#8AFF00", "emoji": "🏦"},
                "profit": {"color": "#8AFF00", "bold": True, "emoji": "💰"},
                "profits": {"color": "#8AFF00", "bold": True, "emoji": "💰"},
                "perte": {"color": "#FF3131", "emoji": "📉"},
                "pertes": {"color": "#FF3131", "emoji": "📉"},
                "échec": {"color": "#FF3131", "emoji": "❌"},
                "echec": {"color": "#FF3131", "emoji": "❌"},
                "budget": {"color": "#FFD700", "emoji": "🧾"},
                "gestion": {"color": "#FFD700", "emoji": "🪙"},
                "roi": {"color": "#8AFF00", "bold": True, "emoji": "📈"},
                "chiffre": {"color": "#FFD700", "emoji": "💰"},
                "ca": {"color": "#FFD700", "emoji": "💰"},
                # Relation & Client
                "client": {"color": "#00E5FF", "underline": True, "emoji": "🤝"},
                "clients": {"color": "#00E5FF", "underline": True, "emoji": "🤝"},
                "collaboration": {"color": "#00E5FF", "emoji": "🫱🏼‍🫲🏽"},
                "collaborations": {"color": "#00E5FF", "emoji": "🫱🏼‍🫲🏽"},
                "communauté": {"color": "#39FF14", "emoji": "🌍"},
                "communaute": {"color": "#39FF14", "emoji": "🌍"},
                "confiance": {"color": "#00E5FF", "emoji": "🔒"},
                "vente": {"color": "#FF73FA", "emoji": "🛒"},
                "ventes": {"color": "#FF73FA", "emoji": "🛒"},
                "deal": {"color": "#FF73FA", "emoji": "📦"},
                "deals": {"color": "#FF73FA", "emoji": "📦"},
                "prospect": {"color": "#00E5FF", "emoji": "🤝"},
                "prospects": {"color": "#00E5FF", "emoji": "🤝"},
                "contrat": {"color": "#FF73FA", "emoji": "📋"},
                # Motivation & Succès
                "succès": {"color": "#39FF14", "italic": True, "emoji": "🏆"},
                "succes": {"color": "#39FF14", "italic": True, "emoji": "🏆"},
                "motivation": {"color": "#FF73FA", "bold": True, "emoji": "🔥"},
                "énergie": {"color": "#FF73FA", "emoji": "⚡"},
                "energie": {"color": "#FF73FA", "emoji": "⚡"},
                "victoire": {"color": "#39FF14", "emoji": "🎯"},
                "discipline": {"color": "#FFD700", "emoji": "⏳"},
                "viral": {"color": "#FF73FA", "bold": True, "emoji": "🚀"},
                "viralité": {"color": "#FF73FA", "bold": True, "emoji": "🌐"},
                "viralite": {"color": "#FF73FA", "bold": True, "emoji": "🌐"},
                "impact": {"color": "#FF73FA", "emoji": "💥"},
                "explose": {"color": "#FF73FA", "emoji": "💥"},
                "explosion": {"color": "#FF73FA", "emoji": "💥"},
                # Risque & Erreurs
                "erreur": {"color": "#FF3131", "emoji": "⚠️"},
                "erreurs": {"color": "#FF3131", "emoji": "⚠️"},
                "warning": {"color": "#FF3131", "emoji": "⚠️"},
                "obstacle": {"color": "#FF3131", "emoji": "🧱"},
                "obstacles": {"color": "#FF3131", "emoji": "🧱"},
                "solution": {"color": "#00E5FF", "emoji": "🔧"},
                "solutions": {"color": "#00E5FF", "emoji": "🔧"},
                "leçon": {"color": "#00E5FF", "emoji": "📚"},
                "lecon": {"color": "#00E5FF", "emoji": "📚"},
                "apprentissage": {"color": "#00E5FF", "emoji": "🧠"},
                "problème": {"color": "#FF3131", "emoji": "🛑"},
                "probleme": {"color": "#FF3131", "emoji": "🛑"},
            }
            add_hormozi_subtitles(
                str(with_broll_path), subtitles, str(final_subtitled_path),
                brand_kit=getattr(Config, 'BRAND_KIT_ID', 'default'),
                span_style_map=span_style_map
            )
        except Exception as e:
            print(f"  ❌ Erreur ajout sous-titres Hormozi: {e}")
            # Pas de retour anticipé: continuer export simple
        
        # Export final accumulé dans output/final/ et sous-titré (burn-in) dans output/subtitled/
        final_dir = Config.OUTPUT_FOLDER / 'final'
        subtitled_dir = Config.OUTPUT_FOLDER / 'subtitled'
        # Noms de base sans extension
        base_name = clip_path.stem
        output_path = self._unique_path(final_dir, f"final_{base_name}", ".mp4")
        try:
            # Choisir source finale: si sous-titrée existe sinon with_broll sinon reframed
            source_final = None
            if final_subtitled_path.exists():
                source_final = final_subtitled_path
            elif with_broll_path and Path(with_broll_path).exists():
                source_final = with_broll_path
            else:
                source_final = reframed_path
            if source_final and Path(source_final).exists():
                self._hardlink_or_copy(source_final, output_path)
                # Ecrire SRT: éviter le doublon si la vidéo finale a déjà les sous-titres incrustés
                is_burned = (final_subtitled_path.exists() and Path(source_final) == Path(final_subtitled_path))
                if not is_burned:
                    srt_out = output_path.with_suffix('.srt')
                    write_srt(subtitles, srt_out)
                    self._hardlink_or_copy(srt_out, per_clip_dir / 'final.srt')
                    # WebVTT
                    try:
                        vtt_out = output_path.with_suffix('.vtt')
                        write_vtt(subtitles, vtt_out)
                    except Exception:
                        pass
                else:
                    # Produire uniquement une SRT dans le dossier du clip, pas à côté du MP4 final
                    try:
                        write_srt(subtitles, per_clip_dir / 'final.srt')
                    except Exception:
                        pass
                # Toujours produire un VTT à côté du final pour compat
                try:
                    vtt_out = output_path.with_suffix('.vtt')
                    write_vtt(subtitles, vtt_out)
                except Exception:
                    pass
                # Copier final dans dossier clip
                self._hardlink_or_copy(output_path, per_clip_dir / 'final.mp4')
                # Si une version sous-titrée burn-in existe, la dupliquer dans output/subtitled/
                if final_subtitled_path.exists():
                    subtitled_out = self._unique_path(subtitled_dir, f"{base_name}_subtitled", ".mp4")
                    self._hardlink_or_copy(final_subtitled_path, subtitled_out)
                # Copier meta.txt à côté du final accumulé
                try:
                    meta_src = per_clip_dir / 'meta.txt'
                    if meta_src.exists():
                        self._hardlink_or_copy(meta_src, output_path.with_suffix('.txt'))
                except Exception:
                    pass
                # Ecrire un JSON récap par clip
                try:
                    # Durée et hash final
                    final_duration = None
                    try:
                        with VideoFileClip(str(output_path)) as vc:
                            final_duration = float(vc.duration)
                    except Exception:
                        final_duration = None
                    media_hash = None
                    try:
                        from src.pipeline.utils import hash_media  # type: ignore
                    except Exception:
                        hash_media = None  # type: ignore
                    if hash_media:
                        try:
                            media_hash = hash_media(str(output_path))
                        except Exception:
                            media_hash = None
                    summary = {
                        'clip': base_name,
                        'final_mp4': str(output_path.resolve()),
                        'final_srt': str(output_path.with_suffix('.srt').resolve()) if (not is_burned) and output_path.with_suffix('.srt').exists() else None,
                        'final_vtt': str(output_path.with_suffix('.vtt').resolve()) if (not is_burned) and output_path.with_suffix('.vtt').exists() else None,
                        'subtitled_mp4': str((subtitled_out.resolve() if final_subtitled_path.exists() else '')) if final_subtitled_path.exists() else None,
                        'meta_txt': str(output_path.with_suffix('.txt').resolve()) if output_path.with_suffix('.txt').exists() else None,
                        'per_clip_dir': str(per_clip_dir.resolve()),
                        'duration_s': final_duration,
                        'media_hash': media_hash,
                        'events': [
                            {
                                'id': getattr(ev, 'id', ev.get('id') if isinstance(ev, dict) else ''),
                                'start_s': float(getattr(ev, 'start_s', ev.get('start_s') if isinstance(ev, dict) else 0.0) or 0.0),
                                'end_s': float(getattr(ev, 'end_s', ev.get('end_s') if isinstance(ev, dict) else 0.0) or 0.0),
                                'media_path': getattr(ev, 'media_path', ev.get('media_path') if isinstance(ev, dict) else ''),
                                'transition': getattr(ev, 'transition', ev.get('transition') if isinstance(ev, dict) else None),
                                'transition_duration': float(getattr(ev, 'transition_duration', ev.get('transition_duration') if isinstance(ev, dict) else 0.0) or 0.0),
                            } for ev in ([] if 'valid_events' not in locals() else valid_events or [])
                        ]
                    }
                    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as jf:
                        json.dump(summary, jf, ensure_ascii=False, indent=2)
                    # JSONL log
                    try:
                        jsonl = (Config.OUTPUT_FOLDER / 'pipeline.log.jsonl')
                        with open(jsonl, 'a', encoding='utf-8') as lf:
                            lf.write(json.dumps(summary, ensure_ascii=False) + '\n')
                    except Exception:
                        pass
                except Exception:
                    pass
                print(f"  📤 Export terminé: {output_path.name}")
                # Nettoyage des intermédiaires pour limiter l'empreinte disque
                self._cleanup_files([
                    with_broll_path if with_broll_path and with_broll_path != output_path else None,
                ])
                return output_path
            else:
                print(f"  ⚠️ Fichier final introuvable")
                return None
        except Exception as e:
            print(f"  ❌ Erreur export: {e}")
            return None

class PremiereProAutomation:
    """
    Classe pour l'automatisation Premiere Pro (optionnelle)
    Utilise ExtendScript pour les utilisateurs avancés
    """
    
    @staticmethod
    def create_jsx_script(clip_path: str, output_path: str) -> str:
        """Génère un script ExtendScript pour Premiere Pro"""
        jsx_script = f'''
        // Script ExtendScript pour Premiere Pro
        var project = app.project;
        
        // Import du clip
        var importOptions = new ImportOptions();
        importOptions.file = new File("{clip_path}");
        var clip = project.importFiles([importOptions.file]);
        
        // Création d'une séquence 9:16
        var sequence = project.createNewSequence("Vertical_Clip", "HDV-1080i25");
        sequence.videoTracks[0].insertClip(clip[0], 0);
        
        // Application de l'effet Auto Reframe (si disponible)
        // Note: Ceci nécessite Premiere Pro 2019 ou plus récent
        
        // Export
        var encoder = app.encoder;
        encoder.encodeSequence(sequence, "{output_path}", "H.264", false);
        '''
        return jsx_script
    
    @staticmethod 
    def run_premiere_script(jsx_script_content: str):
        """Exécute un script ExtendScript dans Premiere Pro"""
        try:
            # Sauvegarde du script temporaire
            script_path = Config.TEMP_FOLDER / "premiere_script.jsx"
            with open(script_path, 'w') as f:
                f.write(jsx_script_content)
            
            import platform
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                subprocess.run([
                    'osascript', '-e',
                    f'tell application "Adobe Premiere Pro" to do script "{script_path}"'
                ], check=True)
            elif system == 'Windows':
                print("⚠️ Exécution ExtendScript automatisée non supportée nativement sous Windows dans ce pipeline.")
                print("   Ouvrez Premiere Pro et exécutez le script manuellement: " + str(script_path))
            else:
                print("⚠️ Plateforme non supportée pour l'exécution automatique de Premiere Pro.")
            
            logger.info("✅ Script Premiere Pro traité (voir message ci-dessus)")
            
        except Exception as e:
            logger.error(f"❌ Erreur Premiere Pro: {e}")
            raise

# Helper: filter noisy prompt terms
STOP_PROMPT_TERMS = {
    'very','really','clear','stuff','thing','things','some','any','ever','so','much','get','got',
    'will','discuss','this','that','these','those','it','its','im','ive','youve','because'
}

