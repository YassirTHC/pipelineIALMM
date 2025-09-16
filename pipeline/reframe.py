"""Reframing utilities extracted from :mod:`video_processor`."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def configure_imagemagick() -> bool:
    """Configure automatiquement ImageMagick pour MoviePy."""
    try:
        import moviepy.config as cfg

        possible_paths = [
            r"C:\\Program Files\\ImageMagick-7.1.2-Q16-HDRI\\magick.exe",
            r"C:\\Program Files\\ImageMagick-7.1.2-Q16\\magick.exe",
            r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe",
            r"C:\\Program Files\\ImageMagick-7.1.1-Q16\\magick.exe",
            r"C:\\Program Files\\ImageMagick-7.1.0-Q16-HDRI\\magick.exe",
            r"C:\\Program Files\\ImageMagick-7.1.0-Q16\\magick.exe",
        ]

        for path in possible_paths:
            if Path(path).exists():
                cfg.change_settings({"IMAGEMAGICK_BINARY": path})
                print(f"✅ ImageMagick configuré: {path}")
                return True

        print("⚠️ ImageMagick non trouvé, utilisation du mode fallback")
        return False

    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"⚠️ Erreur configuration ImageMagick: {exc}")
        return False


class ReframeProcessor:
    """Encapsule la logique de reframe dynamique."""

    def __init__(
        self,
        config,
        mediapipe_available: bool,
        mediapipe_module,
        logger,
        print_func: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self._mediapipe_available = mediapipe_available
        self._mp = mediapipe_module
        self._logger = logger
        self._print = print_func

    def _get_sample_times(self, duration: float, fps: int) -> List[float]:
        if duration <= 10:
            return list(np.arange(0, duration, 1 / fps))
        if duration <= 30:
            return list(np.arange(0, duration, 2 / fps))
        return list(np.arange(0, duration, 4 / fps))

    @staticmethod
    def _smooth_trajectory(x_centers: List[float], window_size: int = 15) -> List[float]:
        window_size = max(window_size, 31)
        if len(x_centers) < window_size:
            kernel = np.ones(min(9, len(x_centers))) / max(1, min(9, len(x_centers)))
            return np.convolve(x_centers, kernel, mode="same").tolist()
        try:
            from scipy.signal import savgol_filter

            smoothed = savgol_filter(x_centers, window_size, 3).tolist()
        except Exception:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(x_centers, kernel, mode="same").tolist()
        alpha = 0.15
        ema: List[float] = []
        last = smoothed[0] if smoothed else 0.5
        for value in smoothed:
            last = (1 - alpha) * last + alpha * value
            ema.append(last)
        return ema

    def _interpolate_trajectory(
        self, x_centers: List[float], sample_times: List[float], duration: float, fps: int
    ) -> List[float]:
        if not x_centers:
            return [0.5] * int(duration * fps)
        target_times = np.arange(0, duration, 1 / fps)
        if len(x_centers) == 1:
            return [x_centers[0]] * len(target_times)
        try:
            return np.interp(target_times, sample_times, x_centers).tolist()
        except Exception:
            return [x_centers[-1]] * len(target_times)

    def _detect_single_frame(self, image_rgb: np.ndarray) -> float:
        mp_pose = self._mp.solutions.pose
        mp_face = self._mp.solutions.face_detection
        h, w = image_rgb.shape[:2]
        with mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
        ) as pose, mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
            pose_results = pose.process(image_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                key_points = [
                    landmarks[mp_pose.PoseLandmark.NOSE],
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                ]
                valid_points = [p.x for p in key_points if p.visibility > 0.5]
                if valid_points:
                    return sum(valid_points) / len(valid_points)
            face_results = face_detection.process(image_rgb)
            if face_results.detections:
                detection = face_results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                return bbox.xmin + bbox.width / 2
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            moments = [cv2.moments(c) for c in contours if cv2.contourArea(c) > 100]
            if moments:
                centroids_x = [m["m10"] / m["m00"] for m in moments if m["m00"] > 0]
                if centroids_x:
                    return sum(centroids_x) / len(centroids_x) / w
        return 0.5

    def _detect_focus_points(self, video: VideoFileClip, fps: int, duration: float) -> List[float]:
        x_centers: List[float] = []
        sample_times = self._get_sample_times(duration, fps)

        if not self._mediapipe_available or self._mp is None:
            for timestamp in sample_times:
                try:
                    frame = video.get_frame(timestamp)
                    x_center = self._detect_single_frame(frame)
                    x_centers.append(x_center)
                except Exception:
                    x_centers.append(0.5)
            return x_centers

        mp_pose = self._mp.solutions.pose
        mp_face = self._mp.solutions.face_detection
        with mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
        ) as pose, mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
            for timestamp in tqdm(sample_times, desc="🔎 IA focus", leave=False):
                try:
                    frame = video.get_frame(timestamp)
                    image_rgb = frame
                    pose_results = pose.process(image_rgb)
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                        key_points = [
                            landmarks[mp_pose.PoseLandmark.NOSE],
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                        ]
                        valid_points = [p.x for p in key_points if p.visibility > 0.5]
                        if valid_points:
                            x_centers.append(sum(valid_points) / len(valid_points))
                            continue
                    face_results = face_detection.process(image_rgb)
                    if face_results.detections:
                        detection = face_results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        x_centers.append(bbox.xmin + bbox.width / 2)
                        continue
                    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        moments = [cv2.moments(c) for c in contours if cv2.contourArea(c) > 100]
                        centroids_x = [m["m10"] / m["m00"] for m in moments if m["m00"] > 0]
                        if centroids_x:
                            x_centers.append(sum(centroids_x) / len(centroids_x) / image_rgb.shape[1])
                            continue
                    x_centers.append(0.5)
                except Exception:
                    x_centers.append(0.5)
        return self._interpolate_trajectory(x_centers, sample_times, duration, fps)

    def reframe_to_vertical(self, clip_path: Path) -> Path:
        """Applique le reframe dynamique sur ``clip_path``."""
        self._logger.info("🎯 Reframe dynamique avec IA (optimisé)")
        self._print("    🎯 Détection IA en cours...")
        video = VideoFileClip(str(clip_path))
        fps = int(video.fps)
        duration = video.duration
        x_centers = self._detect_focus_points(video, fps, duration)
        x_centers_smooth = self._smooth_trajectory(
            x_centers, window_size=min(15, max(5, len(x_centers) // 4))
        )
        frame_index = 0
        applied_x_center_px: Optional[float] = None
        beta = 0.85

        def crop_frame(frame):
            nonlocal frame_index
            nonlocal applied_x_center_px
            h, w, _ = frame.shape
            if frame_index < len(x_centers_smooth):
                x_target_px = x_centers_smooth[frame_index] * w
            else:
                x_target_px = w * 0.5
            frame_index += 1
            if applied_x_center_px is None:
                applied_x_center_px = x_target_px
            shift = x_target_px - applied_x_center_px
            deadband_px = w * 0.003
            if abs(shift) < deadband_px:
                shift = 0.0
            max_shift_px = w * 0.02
            if shift > max_shift_px:
                shift = max_shift_px
            elif shift < -max_shift_px:
                shift = -max_shift_px
            x_clamped = applied_x_center_px + shift
            applied_x_center_px = beta * applied_x_center_px + (1 - beta) * x_clamped
            target_width = self.config.TARGET_WIDTH
            target_height = self.config.TARGET_HEIGHT
            crop_width = int(target_width * h / target_height)
            crop_width = min(crop_width, w)
            if crop_width % 2 != 0:
                crop_width = crop_width - 1 if crop_width > 1 else crop_width + 1
            x1 = int(max(0, min(w - crop_width, applied_x_center_px - crop_width / 2)))
            x2 = x1 + crop_width
            cropped = frame[:, x1:x2]
            final_width = target_width
            final_height = target_height
            if final_width % 2 != 0:
                final_width = final_width - 1 if final_width > 1 else final_width + 1
            if final_height % 2 != 0:
                final_height = final_height - 1 if final_height > 1 else final_height + 1
            return cv2.resize(cropped, (final_width, final_height), interpolation=cv2.INTER_LANCZOS4)

        reframed = video.fl_image(crop_frame)
        output_path = self.config.TEMP_FOLDER / f"reframed_{clip_path.name}"
        try:
            reframed.write_videofile(
                str(output_path),
                fps=fps,
                codec="h264_nvenc",
                audio_codec="aac",
                verbose=False,
                logger=None,
                preset=None,
                ffmpeg_params=[
                    "-rc",
                    "vbr",
                    "-cq",
                    "19",
                    "-b:v",
                    "0",
                    "-maxrate",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ],
            )
        except Exception:
            reframed.write_videofile(
                str(output_path),
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None,
                preset="medium",
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "20"],
            )
        video.close()
        reframed.close()
        self._print("    ✅ Reframe terminé")
        return output_path
