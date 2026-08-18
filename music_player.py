#!/usr/bin/env python3
import os
import sys
import math
import time
import wave
import datetime
from collections import OrderedDict
from dataclasses import dataclass
from typing import TypedDict
import json

import numpy as np
from mutagen.mp3 import MP3
from mutagen import MutagenError

from PyQt6.QtCore import Qt, QDir, QRect, QUrl, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QColor, QPainter, QPen, QPixmap, QFileSystemModel
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeView,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QLabel,
    QSlider,
    QCheckBox,
    QSplitter,
    QAbstractItemView,
    QFileDialog,
    QMessageBox,
)

try:
    import miniaudio
    HAS_MINIAUDIO = True
except ImportError:
    miniaudio = None
    HAS_MINIAUDIO = False


class WaveformAnalysis(TypedDict):
    levels: np.ndarray
    left_levels: np.ndarray
    right_levels: np.ndarray
    levels_hz: int
    bpm_times_ms: np.ndarray
    bpm_values: np.ndarray


class LoudnessMetrics(TypedDict):
    rms: float
    peak: float


def fmt_seconds(seconds: int) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


@dataclass
class Track:
    filepath: str
    title: str
    artist: str
    year: str
    date: str
    duration_s: int
    mtime: float


class WaveformSegmentLoader(QThread):
    segment_partial = pyqtSignal(int, int, object, int)
    segment_final = pyqtSignal(int, int, object, int)
    analysis_ready = pyqtSignal(object, int)
    loudness_ready = pyqtSignal(float, float, int)
    error = pyqtSignal(str, int)

    def __init__(
        self,
        request_id: int,
        filepath: str,
        duration_ms: int,
        start_ms: int,
        end_ms: int,
        target_peaks: int,
        parent=None,
    ):
        super().__init__(parent)
        self.request_id = request_id
        self.filepath = filepath
        self.duration_ms = max(1, duration_ms)
        self.start_ms = max(0, min(self.duration_ms, start_ms))
        self.end_ms = max(self.start_ms + 1, min(self.duration_ms, end_ms))
        self.target_peaks = max(200, target_peaks)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if HAS_MINIAUDIO:
            self._run_miniaudio_segment()
            return
        if self.filepath.lower().endswith(".wav"):
            self._run_wave_segment_fallback()
            return
        self.error.emit("Waveform backend missing: install miniaudio.", self.request_id)
        self.segment_final.emit(self.start_ms, self.end_ms, None, self.request_id)

    def _decode_rate_for_window(self, window_sec: float, target_peaks: int) -> int:
        # Keep decode density aligned with the visible peak density.
        window_sec = max(0.001, window_sec)
        target_peaks = max(200, target_peaks)
        desired_samples_per_peak = 12
        rate = int((target_peaks * desired_samples_per_peak) / window_sec)
        return max(4000, min(48000, rate))

    @staticmethod
    def _to_peaks(samples: np.ndarray, target_peaks: int) -> np.ndarray:
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)
        spp = max(1, samples.size // max(1, target_peaks))
        used = (samples.size // spp) * spp
        if used <= 0:
            peaks = np.abs(samples).astype(np.float32)
        else:
            peaks = np.abs(samples[:used].reshape(-1, spp)).max(axis=1).astype(np.float32)
        pmax = float(peaks.max()) if peaks.size else 0.0
        if pmax > 0:
            peaks /= pmax
        return peaks

    @staticmethod
    def _to_window_max(samples: np.ndarray, target_points: int) -> np.ndarray:
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)
        target_points = max(1, target_points)
        spp = max(1, samples.size // target_points)
        used = (samples.size // spp) * spp
        if used <= 0:
            values = np.abs(samples).astype(np.float32)
        else:
            values = np.abs(samples[:used].reshape(-1, spp)).max(axis=1).astype(np.float32)
        vmax = float(values.max()) if values.size else 0.0
        if vmax > 0:
            values /= vmax
        return values

    def _estimate_local_bpms(self, levels: np.ndarray, levels_hz: int) -> tuple[np.ndarray, np.ndarray]:
        if levels.size < levels_hz * 4:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        smooth_len = max(3, levels_hz // 50)
        smooth_kernel = np.ones(smooth_len, dtype=np.float32) / smooth_len
        smoothed = np.convolve(levels, smooth_kernel, mode="same").astype(np.float32)
        onset = np.maximum(0.0, np.diff(smoothed, prepend=smoothed[0])).astype(np.float32)
        local_mean_len = max(8, levels_hz // 8)
        local_mean_kernel = np.ones(local_mean_len, dtype=np.float32) / local_mean_len
        onset = np.maximum(0.0, onset - np.convolve(onset, local_mean_kernel, mode="same"))
        onset -= float(onset.mean())
        onset_std = float(onset.std())
        if onset_std <= 1e-6:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        onset /= onset_std

        min_bpm = 70.0
        max_bpm = 190.0
        min_lag = max(1, int(levels_hz * 60.0 / max_bpm))
        max_lag = max(min_lag + 1, int(levels_hz * 60.0 / min_bpm))
        window_frames = min(len(onset), max(levels_hz * 6, levels_hz * 10))
        step_frames = max(1, levels_hz // 4)
        if window_frames <= max_lag + 1:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        bpm_times: list[int] = []
        bpm_values: list[float] = []
        window_fn = np.hanning(window_frames).astype(np.float32)
        previous_bpm = 0.0

        def fold_bpm(bpm: float) -> float:
            while bpm < 80.0:
                bpm *= 2.0
            while bpm > 175.0:
                bpm *= 0.5
            return bpm

        def continuity_distance(candidate: float, reference: float) -> float:
            if reference <= 0.0:
                return 0.0
            return min(
                abs(candidate - reference),
                abs(candidate - (reference * 0.5)),
                abs(candidate - (reference * 2.0)),
            )

        for end in range(window_frames, len(onset) + 1, step_frames):
            segment = onset[end - window_frames:end] * window_fn
            corr_values: list[float] = []
            candidate_bpms: list[float] = []
            for lag in range(min_lag, max_lag + 1):
                base = float(np.dot(segment[:-lag], segment[lag:]))
                if base <= 0.0:
                    corr_values.append(0.0)
                    candidate_bpms.append(fold_bpm((60.0 * levels_hz) / lag))
                    continue
                score = base
                half_lag = lag // 2
                double_lag = lag * 2
                if half_lag >= min_lag:
                    score += 0.35 * float(np.dot(segment[:-half_lag], segment[half_lag:]))
                if double_lag < len(segment):
                    score += 0.20 * float(np.dot(segment[:-double_lag], segment[double_lag:]))
                bpm = fold_bpm((60.0 * levels_hz) / lag)
                if previous_bpm > 0.0:
                    score *= max(0.55, 1.0 - (continuity_distance(bpm, previous_bpm) / 90.0))
                corr_values.append(score)
                candidate_bpms.append(bpm)
            corr = np.asarray(corr_values, dtype=np.float32)
            if corr.size == 0:
                continue
            best_indices = np.argsort(corr)[-5:]
            if best_indices.size == 0:
                continue
            best_corr = float(corr[best_indices[-1]])
            if best_corr <= 0.0:
                continue
            weighted_sum = 0.0
            weight_total = 0.0
            for idx in best_indices:
                score = float(corr[int(idx)])
                if score <= 0.0:
                    continue
                bpm = candidate_bpms[int(idx)]
                weighted_sum += bpm * score
                weight_total += score
            if weight_total <= 0.0:
                continue
            bpm = weighted_sum / weight_total
            if bpm_values:
                recent = bpm_values[-4:]
                bpm = float((0.65 * bpm) + (0.35 * float(np.median(recent))))
            previous_bpm = bpm
            bpm_times.append(int(self.start_ms + (((end - (window_frames // 2)) / levels_hz) * 1000.0)))
            bpm_values.append(float(bpm))

        if not bpm_times:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        return np.asarray(bpm_times, dtype=np.int32), np.asarray(bpm_values, dtype=np.float32)

    def _build_analysis_payload(
        self, left_samples: np.ndarray, right_samples: np.ndarray, mono_samples: np.ndarray
    ) -> WaveformAnalysis:
        levels_hz = 200
        level_points = max(1, int(math.ceil(((self.end_ms - self.start_ms) / 1000.0) * levels_hz)))
        levels = self._to_window_max(mono_samples, level_points)
        left_levels = self._to_window_max(left_samples, level_points)
        right_levels = self._to_window_max(right_samples, level_points)
        if levels.size:
            levels = np.sqrt(levels).astype(np.float32, copy=False)
        if left_levels.size:
            left_levels = np.sqrt(left_levels).astype(np.float32, copy=False)
        if right_levels.size:
            right_levels = np.sqrt(right_levels).astype(np.float32, copy=False)
        bpm_times_ms, bpm_values = self._estimate_local_bpms(levels, levels_hz)
        return {
            "levels": levels.astype(np.float32, copy=False),
            "left_levels": left_levels.astype(np.float32, copy=False),
            "right_levels": right_levels.astype(np.float32, copy=False),
            "levels_hz": levels_hz,
            "bpm_times_ms": bpm_times_ms,
            "bpm_values": bpm_values,
        }

    def _emit_from_samples(self, left_samples: np.ndarray, right_samples: np.ndarray):
        if left_samples.size == 0 or right_samples.size == 0:
            self.segment_final.emit(self.start_ms, self.end_ms, None, self.request_id)
            return
        left_peaks = self._to_peaks(left_samples, self.target_peaks)
        right_peaks = self._to_peaks(right_samples, self.target_peaks)
        peaks = (left_peaks, right_peaks)
        mono = 0.5 * (left_samples + right_samples)
        self.analysis_ready.emit(self._build_analysis_payload(left_samples, right_samples, mono), self.request_id)
        rms = float(np.sqrt(np.mean(mono * mono)))
        abs_max = float(np.max(np.abs(mono)))
        if abs_max > 0:
            rms /= abs_max
        self.loudness_ready.emit(rms, abs_max, self.request_id)
        self.segment_final.emit(self.start_ms, self.end_ms, peaks, self.request_id)

    def _run_miniaudio_segment(self):
        try:
            backend = miniaudio
            if backend is None:
                self.error.emit("Waveform backend missing: install miniaudio.", self.request_id)
                self.segment_final.emit(self.start_ms, self.end_ms, None, self.request_id)
                return
            window_sec = max(0.001, (self.end_ms - self.start_ms) / 1000.0)
            decode_rate = self._decode_rate_for_window(window_sec, self.target_peaks)
            start_frame = int((self.start_ms / 1000.0) * decode_rate)
            total_frames = int(window_sec * decode_rate)
            read_frames = 0
            chunk_frames = max(2048, decode_rate // 2)
            collected_left: list[np.ndarray] = []
            collected_right: list[np.ndarray] = []

            stream = backend.stream_file(
                self.filepath,
                output_format=backend.SampleFormat.SIGNED16,
                nchannels=2,
                sample_rate=decode_rate,
                frames_to_read=chunk_frames,
                seek_frame=start_frame,
            )

            for raw in stream:
                if self._cancelled:
                    return
                if read_frames >= total_frames:
                    break
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                if arr.size == 0:
                    continue
                if arr.size % 2 != 0:
                    arr = arr[:-1]
                if arr.size == 0:
                    continue
                stereo = arr.reshape(-1, 2)
                left = stereo[:, 0]
                right = stereo[:, 1]
                remaining = total_frames - read_frames
                take = min(len(left), remaining)
                if take <= 0:
                    break
                left = left[:take]
                right = right[:take]
                collected_left.append(left)
                collected_right.append(right)
                read_frames += take

            if self._cancelled:
                return
            if not collected_left:
                self.error.emit("No audio data for selected segment.", self.request_id)
                self.segment_final.emit(self.start_ms, self.end_ms, None, self.request_id)
                return
            self._emit_from_samples(np.concatenate(collected_left), np.concatenate(collected_right))
        except Exception as exc:
            self.error.emit(f"Waveform decode error: {exc}", self.request_id)
            self.segment_final.emit(self.start_ms, self.end_ms, None, self.request_id)

    def _run_wave_segment_fallback(self):
        try:
            with wave.open(self.filepath, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sw = wf.getsampwidth()
                start_frame = int((self.start_ms / 1000.0) * sr)
                end_frame = int((self.end_ms / 1000.0) * sr)
                total_frames = max(1, end_frame - start_frame)
                wf.setpos(min(start_frame, wf.getnframes() - 1))
                chunk_frames = max(2048, sr // 2)
                read_frames = 0
                collected_left: list[np.ndarray] = []
                collected_right: list[np.ndarray] = []

                while read_frames < total_frames:
                    if self._cancelled:
                        return
                    need = min(chunk_frames, total_frames - read_frames)
                    raw = wf.readframes(need)
                    if not raw:
                        break
                    if sw == 2:
                        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    else:
                        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
                    if arr.size == 0:
                        break
                    if channels > 1:
                        arr = arr.reshape(-1, channels)
                        left = arr[:, 0]
                        right = arr[:, 1] if channels > 1 else arr[:, 0]
                    else:
                        left = arr
                        right = arr
                    collected_left.append(left.astype(np.float32))
                    collected_right.append(right.astype(np.float32))
                    read_frames += len(left)

                if self._cancelled:
                    return
                if not collected_left:
                    self.error.emit("No WAV data for selected segment.", self.request_id)
                    self.segment_final.emit(self.start_ms, self.end_ms, None, self.request_id)
                    return
                self._emit_from_samples(np.concatenate(collected_left), np.concatenate(collected_right))
        except Exception as exc:
            self.error.emit(f"WAV waveform decode error: {exc}", self.request_id)
            self.segment_final.emit(self.start_ms, self.end_ms, None, self.request_id)


class WaveformView(QWidget):
    seek_preview = pyqtSignal(int)
    seek_commit = pyqtSignal(int)
    zoom_changed = pyqtSignal(int)
    viewport_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._peaks: tuple[np.ndarray, np.ndarray] | None = None
        self._segment_start_ms = 0
        self._segment_end_ms = 0
        self._duration_ms = 0
        self._position_ms = 0
        self._status = "No waveform"
        self._loading = False
        self._drag_seek = False
        self._drag_pan = False
        self._pan_start_x = 0
        self._pan_start_range = (0, 0)

        self._view_start_ms = 0
        self._view_end_ms = 0
        self._zoom_level = 1
        self._hover_x = -1
        self._default_window_limit_ms = 10 * 1000
        self._edge_zone_px = 20
        self._edge_speed_factor = 0.0
        self._edge_direction = 0
        self._edge_anchor_x = 0
        self._edge_scroll_timer = QTimer(self)
        self._edge_scroll_timer.setInterval(16)
        self._edge_scroll_timer.timeout.connect(self._tick_edge_scroll)
        self._follow_playback = False
        self._paint_hz = 0.0
        self._last_paint_ts = 0.0
        self._levels = np.empty(0, dtype=np.float32)
        self._left_levels = np.empty(0, dtype=np.float32)
        self._right_levels = np.empty(0, dtype=np.float32)
        self._levels_hz = 0
        self._bpm_times_ms = np.empty(0, dtype=np.int32)
        self._bpm_values = np.empty(0, dtype=np.float32)
        self._current_level = 0.0
        self._current_left_level = 0.0
        self._current_right_level = 0.0
        self._display_left_level = 0.0
        self._display_right_level = 0.0
        self._peak_hold_left = 0.0
        self._peak_hold_right = 0.0
        self._peak_hold_last_ts = time.perf_counter()
        self._current_bpm = 0.0
        self._paint_cache_key: tuple[int, int, int, int, int, int] | None = None
        self._paint_left_cache = np.empty(0, dtype=np.float32)
        self._paint_right_cache = np.empty(0, dtype=np.float32)
        self._static_layer_key: tuple[int, int, int, int, int, int, int, str] | None = None
        self._static_layer_pixmap: QPixmap | None = None
        self._wave_layer_key: tuple[int, int, int, int, int, int, int] | None = None
        self._wave_layer_pixmap: QPixmap | None = None

        self.setMinimumHeight(100)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

    def _invalidate_paint_cache(self):
        self._paint_cache_key = None
        self._paint_left_cache = np.empty(0, dtype=np.float32)
        self._paint_right_cache = np.empty(0, dtype=np.float32)
        self._invalidate_static_layer()
        self._invalidate_wave_layer()

    def _invalidate_static_layer(self):
        self._static_layer_key = None
        self._static_layer_pixmap = None

    def _invalidate_wave_layer(self):
        self._wave_layer_key = None
        self._wave_layer_pixmap = None

    @staticmethod
    def _union_rect(base: QRect | None, extra: QRect) -> QRect:
        return extra if base is None else base.united(extra)

    def _hud_height(self) -> int:
        return 36

    def _minimap_height(self) -> int:
        return 8

    def _wave_rect(self) -> QRect:
        hud_h = self._hud_height()
        mini_h = self._minimap_height()
        draw_h = max(24, self.height() - mini_h - hud_h)
        return QRect(0, hud_h, self.width(), draw_h)

    def _hud_rect(self) -> QRect:
        return QRect(0, 0, self.width(), self._hud_height())

    def _line_dirty_rect(self, x: int, top: int, bottom: int, thickness: int = 4) -> QRect | None:
        if x < 0:
            return None
        width = self.width()
        if width <= 0 or bottom <= top:
            return None
        left = max(0, x - thickness)
        right = min(width, x + thickness + 1)
        if right <= left:
            return None
        return QRect(left, top, right - left, bottom - top)

    def _request_overlay_update(
        self,
        prev_playhead_x: int = -1,
        new_playhead_x: int = -1,
        prev_hover_x: int = -1,
        new_hover_x: int = -1,
        hud_changed: bool = False,
        full: bool = False,
    ):
        if full:
            self.update()
            return
        dirty: QRect | None = self._hud_rect() if hud_changed else None
        wave_rect = self._wave_rect()
        wave_top = wave_rect.y()
        wave_bottom = wave_top + wave_rect.height()
        for x in (prev_playhead_x, new_playhead_x, prev_hover_x, new_hover_x):
            rect = self._line_dirty_rect(x, wave_top, wave_bottom)
            if rect is not None:
                dirty = self._union_rect(dirty, rect)
        if dirty is not None:
            self.update(dirty)

    def _get_paint_arrays(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        if self._peaks is None or self._duration_ms <= 0 or width <= 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        left_peaks, right_peaks = self._peaks
        n = min(len(left_peaks), len(right_peaks))
        if n <= 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        key = (
            width,
            self._view_start_ms,
            self._view_end_ms,
            self._segment_start_ms,
            self._segment_end_ms,
            n,
        )
        if self._paint_cache_key == key:
            return self._paint_left_cache, self._paint_right_cache

        seg_span = max(1, self._segment_end_ms - self._segment_start_ms)
        view_span = max(1, self._view_end_ms - self._view_start_ms)
        times = self._view_start_ms + np.linspace(0.0, view_span, num=width, dtype=np.float32)
        mask = (times >= self._segment_start_ms) & (times <= self._segment_end_ms)
        sampled_left = np.zeros(width, dtype=np.float32)
        sampled_right = np.zeros(width, dtype=np.float32)
        if np.any(mask):
            ratios = (times[mask] - self._segment_start_ms) / seg_span
            indices = np.clip((ratios * (n - 1)).astype(np.int32), 0, n - 1)
            sampled_left[mask] = left_peaks[indices]
            sampled_right[mask] = right_peaks[indices]
        self._paint_cache_key = key
        self._paint_left_cache = sampled_left
        self._paint_right_cache = sampled_right
        return sampled_left, sampled_right

    @staticmethod
    def _draw_wave_columns(
        p: QPainter,
        left_render: np.ndarray,
        right_render: np.ndarray,
        x_start: int,
        x_end: int,
        height: int,
    ):
        if x_end <= x_start:
            return
        center = height // 2
        wave_half = max(1, center - 3)
        blue = QPen(QColor(40, 120, 255, 110), 1)
        yellow = QPen(QColor(255, 220, 70, 110), 1)
        green = QPen(QColor(80, 230, 90, 150), 1)
        for x in range(x_start, min(x_end, len(left_render), len(right_render))):
            lv = float(left_render[x])
            rv = float(right_render[x])
            if lv <= 0.0 and rv <= 0.0:
                continue
            lamp = max(1, int(lv * wave_half))
            ramp = max(1, int(rv * wave_half))
            oamp = min(lamp, ramp)
            p.setPen(blue)
            p.drawLine(x, center - lamp, x, center + lamp)
            p.setPen(yellow)
            p.drawLine(x, center - ramp, x, center + ramp)
            p.setPen(green)
            p.drawLine(x, center - oamp, x, center + oamp)

    def _ensure_wave_layer(self, width: int, height: int):
        if self._peaks is None or self._duration_ms <= 0 or width <= 0 or height <= 0:
            self._wave_layer_key = None
            self._wave_layer_pixmap = None
            return
        left_render, right_render = self._get_paint_arrays(width)
        n = min(len(left_render), len(right_render))
        key = (
            width,
            height,
            self._view_start_ms,
            self._view_end_ms,
            self._segment_start_ms,
            self._segment_end_ms,
            n,
        )
        if self._wave_layer_key == key and self._wave_layer_pixmap is not None:
            return
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(0, 0, 0, 0))
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._draw_wave_columns(p, left_render, right_render, 0, width, height)
        p.end()
        self._wave_layer_key = key
        self._wave_layer_pixmap = pixmap

    def _try_shift_wave_layer(self, prev_start: int, prev_end: int) -> bool:
        wave_rect = self._wave_rect()
        width = wave_rect.width()
        height = wave_rect.height()
        if (
            self._wave_layer_pixmap is None
            or self._wave_layer_key is None
            or self._peaks is None
            or width <= 0
            or height <= 0
        ):
            return False
        prev_span = max(1, prev_end - prev_start)
        new_span = max(1, self._view_end_ms - self._view_start_ms)
        if prev_span != new_span:
            return False
        ms_per_px = prev_span / max(1, width)
        shift_px = int(round((prev_start - self._view_start_ms) / ms_per_px))
        if shift_px == 0 or abs(shift_px) >= width:
            return False

        left_render, right_render = self._get_paint_arrays(width)
        n = min(len(left_render), len(right_render))
        new_pixmap = QPixmap(width, height)
        new_pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(new_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.drawPixmap(shift_px, 0, self._wave_layer_pixmap)
        if shift_px < 0:
            self._draw_wave_columns(painter, left_render, right_render, width + shift_px, width, height)
        else:
            self._draw_wave_columns(painter, left_render, right_render, 0, shift_px, height)
        painter.end()
        self._wave_layer_pixmap = new_pixmap
        self._wave_layer_key = (
            width,
            height,
            self._view_start_ms,
            self._view_end_ms,
            self._segment_start_ms,
            self._segment_end_ms,
            n,
        )
        return True

    def _ensure_static_layer(self, width: int, height: int, hud_h: int, mini_h: int):
        key = (
            width,
            height,
            hud_h,
            mini_h,
            self._view_start_ms,
            self._view_end_ms,
            self._duration_ms,
            self._status,
        )
        if self._static_layer_key == key and self._static_layer_pixmap is not None:
            return

        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(16, 16, 16))
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        draw_top = hud_h
        draw_h = max(24, height - mini_h - draw_top)
        center = draw_top + (draw_h // 2)

        p.fillRect(0, 0, width, hud_h, QColor(10, 10, 10))
        p.setPen(QPen(QColor(42, 42, 42), 1))
        p.drawLine(0, hud_h - 1, width, hud_h - 1)
        p.drawLine(0, center, width, center)

        if self._peaks is None or self._duration_ms <= 0:
            if self._status:
                p.setPen(QPen(QColor(120, 120, 120), 1))
                p.drawText(0, draw_top, width, draw_h, Qt.AlignmentFlag.AlignCenter, self._status)

        self._draw_time_scale(p, width, draw_top)
        self._draw_minimap(p, width, height, mini_h)
        p.end()
        self._static_layer_key = key
        self._static_layer_pixmap = pixmap

    def set_default_window_limit_ms(self, limit_ms: int):
        self._default_window_limit_ms = max(0, limit_ms)

    def set_follow_playback(self, enabled: bool):
        """Enable/disable follow-playback mode (playhead centered, waveform scrolls)."""
        self._follow_playback = enabled

    def get_viewport_ms(self) -> tuple[int, int]:
        return self._view_start_ms, self._view_end_ms

    def get_zoom_level(self) -> int:
        return self._zoom_level

    def get_segment_data(self):
        if self._peaks is None:
            return None
        return self._segment_start_ms, self._segment_end_ms, self._peaks

    def set_segment_data(self, start_ms: int, end_ms: int, peaks, loading: bool = False):
        self._segment_start_ms = start_ms
        self._segment_end_ms = end_ms
        self._peaks = peaks
        self._loading = loading
        self._invalidate_paint_cache()
        if peaks is not None:
            self._status = ""
        self._request_overlay_update(full=True)

    def preview_stretch_to_view(self, start_ms: int, end_ms: int, target_peaks: int):
        seg = self.get_segment_data()
        if seg is None:
            return
        seg_start, seg_end, seg_peaks = seg
        if seg_peaks is None:
            return
        left, right = seg_peaks
        if len(left) == 0 or len(right) == 0:
            return

        out_l = np.zeros(max(1, target_peaks), dtype=np.float32)
        out_r = np.zeros(max(1, target_peaks), dtype=np.float32)
        seg_span = max(1, seg_end - seg_start)
        view_span = max(1, end_ms - start_ms)
        for i in range(len(out_l)):
            t = start_ms + int((i / max(1, len(out_l) - 1)) * view_span)
            if t < seg_start or t > seg_end:
                continue
            r = (t - seg_start) / seg_span
            idx = int(max(0.0, min(1.0, r)) * (len(left) - 1))
            out_l[i] = float(left[idx])
            out_r[i] = float(right[idx])
        self.set_segment_data(start_ms, end_ms, (out_l, out_r), loading=True)

    def set_track_duration(self, duration_ms: int):
        self._duration_ms = max(0, duration_ms)
        self._set_default_initial_view()
        self._invalidate_paint_cache()
        self.zoom_changed.emit(self._zoom_level)
        self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
        self._request_overlay_update(full=True)

    def clear_analysis(self):
        self._levels = np.empty(0, dtype=np.float32)
        self._left_levels = np.empty(0, dtype=np.float32)
        self._right_levels = np.empty(0, dtype=np.float32)
        self._levels_hz = 0
        self._bpm_times_ms = np.empty(0, dtype=np.int32)
        self._bpm_values = np.empty(0, dtype=np.float32)
        self._current_level = 0.0
        self._current_left_level = 0.0
        self._current_right_level = 0.0
        self._display_left_level = 0.0
        self._display_right_level = 0.0
        self._peak_hold_left = 0.0
        self._peak_hold_right = 0.0
        self._peak_hold_last_ts = time.perf_counter()
        self._current_bpm = 0.0
        self._request_overlay_update(full=True)

    def set_analysis_data(self, analysis: WaveformAnalysis | None):
        if not analysis:
            self.clear_analysis()
            return
        self._levels = np.asarray(analysis["levels"], dtype=np.float32)
        self._left_levels = np.asarray(analysis["left_levels"], dtype=np.float32)
        self._right_levels = np.asarray(analysis["right_levels"], dtype=np.float32)
        self._levels_hz = int(analysis["levels_hz"])
        self._bpm_times_ms = np.asarray(analysis["bpm_times_ms"], dtype=np.int32)
        self._bpm_values = np.asarray(analysis["bpm_values"], dtype=np.float32)
        self._update_analysis_position()
        self.update()

    def _set_default_initial_view(self):
        if self._duration_ms <= 0:
            self._view_start_ms = 0
            self._view_end_ms = 0
            self._zoom_level = 1
            return
        default_window_ms = self._duration_ms
        if self._default_window_limit_ms > 0:
            default_window_ms = min(self._duration_ms, self._default_window_limit_ms)
        self._view_start_ms = 0
        self._view_end_ms = default_window_ms
        self._zoom_level = self._zoom_level_for_span(default_window_ms)

    def set_loading(self):
        self._loading = True
        self._status = "Loading waveform ..."
        self.clear_analysis()
        self._invalidate_paint_cache()
        self._request_overlay_update(full=True)

    def clear_for_view(self, start_ms: int, end_ms: int):
        self._segment_start_ms = start_ms
        self._segment_end_ms = end_ms
        self._peaks = None
        self._loading = True
        self._status = "Loading waveform ..."
        self._invalidate_paint_cache()
        self.clear_analysis()
        self._request_overlay_update(full=True)

    def set_error(self, msg: str):
        self._loading = False
        if self._peaks is None:
            self._status = msg
        self._invalidate_paint_cache()
        self._request_overlay_update(full=True)

    def set_partial(self, start_ms: int, end_ms: int, peaks):
        if peaks is None:
            return
        self._segment_start_ms = start_ms
        self._segment_end_ms = end_ms
        self._peaks = peaks
        self._loading = True
        self._status = ""
        self.update()

    def set_final(self, start_ms: int, end_ms: int, peaks):
        self._segment_start_ms = start_ms
        self._segment_end_ms = end_ms
        self._peaks = peaks
        self._loading = False
        self._invalidate_paint_cache()
        if peaks is None:
            self._status = "Waveform unavailable"
        else:
            self._status = ""
        self._request_overlay_update(full=True)

    def set_position(self, ms: int):
        prev_x = self._ms_to_x(self._position_ms) if self._duration_ms > 0 else -1
        prev_hover_x = self._hover_x
        prev_left = self._display_left_level
        prev_right = self._display_right_level
        prev_peak_left = self._peak_hold_left
        prev_peak_right = self._peak_hold_right
        prev_bpm = self._current_bpm
        prev_view = (self._view_start_ms, self._view_end_ms)
        self._position_ms = max(0, ms)
        self._update_analysis_position()
        viewport_moved = False
        if not self._drag_seek:
            viewport_moved = self._auto_scroll()
            if viewport_moved and not self._follow_playback:
                self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
        if viewport_moved or prev_view != (self._view_start_ms, self._view_end_ms):
            self._invalidate_paint_cache()
        new_x = self._ms_to_x(self._position_ms) if self._duration_ms > 0 else -1
        if (
            not viewport_moved
            and prev_x == new_x
            and abs(prev_left - self._display_left_level) < 0.002
            and abs(prev_right - self._display_right_level) < 0.002
            and abs(prev_peak_left - self._peak_hold_left) < 0.002
            and abs(prev_peak_right - self._peak_hold_right) < 0.002
            and abs(prev_bpm - self._current_bpm) < 0.05
        ):
            return
        self._request_overlay_update(
            prev_playhead_x=prev_x,
            new_playhead_x=new_x,
            prev_hover_x=prev_hover_x,
            new_hover_x=self._hover_x,
            hud_changed=(
                abs(prev_left - self._display_left_level) >= 0.002
                or abs(prev_right - self._display_right_level) >= 0.002
                or abs(prev_peak_left - self._peak_hold_left) >= 0.002
                or abs(prev_peak_right - self._peak_hold_right) >= 0.002
                or abs(prev_bpm - self._current_bpm) >= 0.05
            ),
            full=viewport_moved or prev_view != (self._view_start_ms, self._view_end_ms),
        )

    def _update_analysis_position(self):
        now = time.perf_counter()
        decay_dt = max(0.0, now - self._peak_hold_last_ts)
        self._peak_hold_last_ts = now
        self._current_level = 0.0
        self._current_left_level = 0.0
        self._current_right_level = 0.0
        if self._levels_hz > 0 and self._levels.size:
            idx = int((self._position_ms / 1000.0) * self._levels_hz)
            idx = max(0, min(len(self._levels) - 1, idx))
            self._current_level = float(self._levels[idx])
            if self._left_levels.size:
                self._current_left_level = float(self._left_levels[idx])
            if self._right_levels.size:
                self._current_right_level = float(self._right_levels[idx])
        # Smooth the visible meters so they feel calmer than the raw analysis stream.
        attack_per_second = 4.5
        release_per_second = 1.4
        left_rate = attack_per_second if self._current_left_level > self._display_left_level else release_per_second
        right_rate = attack_per_second if self._current_right_level > self._display_right_level else release_per_second
        left_mix = min(1.0, decay_dt * left_rate)
        right_mix = min(1.0, decay_dt * right_rate)
        self._display_left_level += (self._current_left_level - self._display_left_level) * left_mix
        self._display_right_level += (self._current_right_level - self._display_right_level) * right_mix
        if self._display_left_level < 0.008:
            self._display_left_level = 0.0
        if self._display_right_level < 0.008:
            self._display_right_level = 0.0
        decay_per_second = 0.16
        self._peak_hold_left = max(0.0, self._peak_hold_left - (decay_per_second * decay_dt))
        self._peak_hold_right = max(0.0, self._peak_hold_right - (decay_per_second * decay_dt))
        self._peak_hold_left = max(self._peak_hold_left, self._current_left_level, self._display_left_level)
        self._peak_hold_right = max(self._peak_hold_right, self._current_right_level, self._display_right_level)

        self._current_bpm = 0.0
        if self._bpm_times_ms.size and self._bpm_values.size:
            idx = int(np.searchsorted(self._bpm_times_ms, self._position_ms, side="right") - 1)
            if 0 <= idx < len(self._bpm_values):
                start_idx = max(0, idx - 3)
                recent = self._bpm_values[start_idx:idx + 1]
                self._current_bpm = float(np.median(recent))

    def set_zoom_level(self, level: int):
        level = max(1, min(100, int(level)))
        self._zoom_level = level
        self._apply_zoom_around_center()
        self._invalidate_paint_cache()
        self.zoom_changed.emit(level)
        self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
        self._request_overlay_update(full=True)

    def reset_zoom(self):
        self._zoom_level = 1
        self._set_default_initial_view()
        self._invalidate_paint_cache()
        self.zoom_changed.emit(self._zoom_level)
        self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
        self._request_overlay_update(full=True)

    def wheelEvent(self, a0):
        assert a0 is not None
        if a0.angleDelta().y() > 0:
            self.set_zoom_level(self._zoom_level + 2)
        else:
            self.set_zoom_level(self._zoom_level - 2)
        a0.accept()

    def mouseDoubleClickEvent(self, a0):
        assert a0 is not None
        self.reset_zoom()
        a0.accept()

    def mousePressEvent(self, a0):
        assert a0 is not None
        if a0.button() == Qt.MouseButton.RightButton:
            self._drag_pan = True
            self._pan_start_x = int(a0.position().x())
            self._pan_start_range = (self._view_start_ms, self._view_end_ms)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if a0.button() == Qt.MouseButton.LeftButton:
            self._drag_seek = True
            self._edge_anchor_x = int(a0.position().x())
            self._update_edge_scroll_state(self._edge_anchor_x)
            ms = self._x_to_ms(a0.position().x())
            self.seek_preview.emit(ms)
            self._request_overlay_update(
                prev_playhead_x=self._ms_to_x(self._position_ms),
                new_playhead_x=self._ms_to_x(ms),
                prev_hover_x=self._hover_x,
                new_hover_x=self._hover_x,
            )

    def mouseMoveEvent(self, a0):
        assert a0 is not None
        prev_hover_x = self._hover_x
        prev_pos_x = self._ms_to_x(self._position_ms) if self._duration_ms > 0 else -1
        x = int(a0.position().x())
        self._hover_x = x
        if self._drag_seek:
            self._edge_anchor_x = x
            self._update_edge_scroll_state(x)
            ms = self._x_to_ms(x)
            self.seek_preview.emit(ms)
            self._position_ms = ms
        elif self._drag_pan and self._duration_ms > 0:
            dx = int(a0.position().x()) - self._pan_start_x
            span = max(1, self._pan_start_range[1] - self._pan_start_range[0])
            ms_per_px = span / max(1, self.width())
            shift = int(dx * ms_per_px)
            start = self._pan_start_range[0] - shift
            end = self._pan_start_range[1] - shift
            self._set_view_window(start, end)
            self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
            self._update_edge_scroll_state(-1)
            self._request_overlay_update(full=True)
            return
        else:
            self._update_edge_scroll_state(-1)
        self._request_overlay_update(
            prev_playhead_x=prev_pos_x,
            new_playhead_x=self._ms_to_x(self._position_ms) if self._duration_ms > 0 else -1,
            prev_hover_x=prev_hover_x,
            new_hover_x=self._hover_x,
        )

    def mouseReleaseEvent(self, a0):
        assert a0 is not None
        if a0.button() == Qt.MouseButton.RightButton and self._drag_pan:
            self._drag_pan = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if a0.button() == Qt.MouseButton.LeftButton and self._drag_seek:
            prev_pos_x = self._ms_to_x(self._position_ms) if self._duration_ms > 0 else -1
            self._drag_seek = False
            self._update_edge_scroll_state(-1)
            ms = self._position_ms
            self.seek_commit.emit(ms)
            self._request_overlay_update(
                prev_playhead_x=prev_pos_x,
                new_playhead_x=self._ms_to_x(ms) if self._duration_ms > 0 else -1,
                prev_hover_x=self._hover_x,
                new_hover_x=self._hover_x,
            )

    def leaveEvent(self, a0):
        prev_hover_x = self._hover_x
        self._hover_x = -1
        if not self._drag_seek:
            self._update_edge_scroll_state(-1)
        self._request_overlay_update(prev_hover_x=prev_hover_x, new_hover_x=-1)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self._invalidate_paint_cache()
        if self._duration_ms > 0:
            self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
        self._request_overlay_update(full=True)

    def _min_window_ms(self) -> int:
        if self._duration_ms <= 0:
            return 5000
        return max(5000, int(self._duration_ms * 0.002))

    def _window_for_zoom(self) -> int:
        if self._duration_ms <= 0:
            return 0
        min_window = self._min_window_ms()
        exponent = (self._zoom_level - 1) / 99.0
        span = int(round(self._duration_ms * ((min_window / self._duration_ms) ** exponent)))
        return max(min_window, min(self._duration_ms, span))

    def _zoom_level_for_span(self, span_ms: int) -> int:
        if self._duration_ms <= 0:
            return 1
        min_window = self._min_window_ms()
        span = max(min_window, min(self._duration_ms, span_ms))
        if min_window >= self._duration_ms:
            return 1
        a = min_window / self._duration_ms
        b = span / self._duration_ms
        exponent = math.log(b) / math.log(a)
        level = int(round(1 + exponent * 99))
        return max(1, min(100, level))

    def _apply_zoom_around_center(self):
        if self._duration_ms <= 0:
            self._view_start_ms = 0
            self._view_end_ms = 0
            return
        span = self._window_for_zoom()
        if span >= self._duration_ms:
            self._view_start_ms = 0
            self._view_end_ms = self._duration_ms
            return
        center = (self._view_start_ms + self._view_end_ms) // 2
        start = center - span // 2
        end = start + span
        self._set_view_window(start, end)

    def _set_view_window(self, start: int, end: int):
        if self._duration_ms <= 0:
            self._view_start_ms = 0
            self._view_end_ms = 0
            return
        prev_start = self._view_start_ms
        prev_end = self._view_end_ms
        span = max(1, end - start)
        max_start = max(0, self._duration_ms - span)
        start = max(0, min(max_start, start))
        end = start + span
        self._view_start_ms = start
        self._view_end_ms = min(self._duration_ms, end)
        if (prev_start, prev_end) != (self._view_start_ms, self._view_end_ms):
            self._invalidate_static_layer()
            if not self._try_shift_wave_layer(prev_start, prev_end):
                self._paint_cache_key = None
                self._paint_left_cache = np.empty(0, dtype=np.float32)
                self._paint_right_cache = np.empty(0, dtype=np.float32)
                self._invalidate_wave_layer()

    def _quantize_follow_start(self, start: int, span: int) -> int:
        width = max(1, self.width())
        ms_per_px = max(1.0, span / width)
        return int(round(start / ms_per_px) * ms_per_px)

    def _auto_scroll(self) -> bool:
        if self._duration_ms <= 0:
            return False
        moved = False
        
        # Follow-playback mode: center playhead, scroll waveform
        if self._follow_playback:
            span = max(1, self._view_end_ms - self._view_start_ms)
            if span >= self._duration_ms:
                # View covers entire track
                prev = (self._view_start_ms, self._view_end_ms)
                self._view_start_ms = 0
                self._view_end_ms = self._duration_ms
                return prev != (self._view_start_ms, self._view_end_ms)
            # Center playhead in the view
            center = self._position_ms
            start = self._quantize_follow_start(center - span // 2, span)
            prev = (self._view_start_ms, self._view_end_ms)
            self._set_view_window(start, start + span)
            return prev != (self._view_start_ms, self._view_end_ms)
        
        # Normal scroll-to-keep-in-view mode
        if self._zoom_level <= 1:
            prev = (self._view_start_ms, self._view_end_ms)
            self._view_start_ms = 0
            self._view_end_ms = self._duration_ms
            return prev != (self._view_start_ms, self._view_end_ms)
        if self._position_ms < self._view_start_ms or self._position_ms > self._view_end_ms:
            span = max(1, self._view_end_ms - self._view_start_ms)
            start = self._position_ms - span // 2
            prev = (self._view_start_ms, self._view_end_ms)
            self._set_view_window(start, start + span)
            moved = prev != (self._view_start_ms, self._view_end_ms)
        return moved

    def _update_edge_scroll_state(self, x: int):
        if not self._drag_seek or self._duration_ms <= 0:
            self._edge_direction = 0
            self._edge_speed_factor = 0.0
            self._edge_scroll_timer.stop()
            return

        w = max(1, self.width())
        if x < 0 or x > w:
            # outside widget: force strong scrolling in direction
            self._edge_direction = -1 if x < 0 else 1
            self._edge_speed_factor = 1.0
            if not self._edge_scroll_timer.isActive():
                self._edge_scroll_timer.start()
            return

        left_dist = x
        right_dist = w - x
        zone = self._edge_zone_px
        if left_dist <= zone:
            self._edge_direction = -1
            self._edge_speed_factor = max(0.0, 1.0 - (left_dist / max(1, zone)))
        elif right_dist <= zone:
            self._edge_direction = 1
            self._edge_speed_factor = max(0.0, 1.0 - (right_dist / max(1, zone)))
        else:
            self._edge_direction = 0
            self._edge_speed_factor = 0.0

        if self._edge_direction != 0:
            if not self._edge_scroll_timer.isActive():
                self._edge_scroll_timer.start()
        else:
            self._edge_scroll_timer.stop()

    def _tick_edge_scroll(self):
        if not self._drag_seek or self._edge_direction == 0 or self._duration_ms <= 0:
            self._edge_scroll_timer.stop()
            return
        span = max(1, self._view_end_ms - self._view_start_ms)
        base_step = max(10, span // 250)  # baseline
        accel = 1.0 + (self._edge_speed_factor ** 2) * 18.0
        step_ms = int(base_step * accel)
        if step_ms <= 0:
            return
        prev = (self._view_start_ms, self._view_end_ms)
        self._set_view_window(
            self._view_start_ms + (self._edge_direction * step_ms),
            self._view_end_ms + (self._edge_direction * step_ms),
        )
        moved = prev != (self._view_start_ms, self._view_end_ms)
        if moved:
            self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
            self._position_ms = self._x_to_ms(self._edge_anchor_x)
            self.seek_preview.emit(self._position_ms)
            self._request_overlay_update(full=True)
    def _x_to_ms(self, x: float) -> int:
        if self._duration_ms <= 0:
            return 0
        ratio = max(0.0, min(1.0, float(x) / max(1, self.width())))
        span = max(1, self._view_end_ms - self._view_start_ms)
        return self._view_start_ms + int(span * ratio)

    def _ms_to_x(self, ms: int) -> int:
        span = max(1, self._view_end_ms - self._view_start_ms)
        ratio = (ms - self._view_start_ms) / span
        return int(max(0.0, min(1.0, ratio)) * self.width())

    def paintEvent(self, a0):
        now = time.perf_counter()
        if self._last_paint_ts > 0.0:
            delta = now - self._last_paint_ts
            if delta > 0.0:
                inst_hz = 1.0 / delta
                self._paint_hz = inst_hz if self._paint_hz <= 0.0 else (self._paint_hz * 0.85) + (inst_hz * 0.15)
        self._last_paint_ts = now

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w = self.width()
        h = self.height()
        hud_h = 36
        mini_h = 8
        draw_top = hud_h
        draw_h = max(24, h - mini_h - draw_top)
        draw_bottom = draw_top + draw_h
        self._ensure_static_layer(w, h, hud_h, mini_h)
        if self._static_layer_pixmap is not None:
            p.drawPixmap(0, 0, self._static_layer_pixmap)
        self._ensure_wave_layer(w, draw_h)
        if self._wave_layer_pixmap is not None:
            p.drawPixmap(0, draw_top, self._wave_layer_pixmap)

        if self._duration_ms > 0:
            pos_x = self._ms_to_x(self._position_ms)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(pos_x, draw_top, pos_x, draw_bottom - 1)

        if self._hover_x >= 0 and not self._drag_pan and not self._drag_seek:
            p.setPen(QPen(QColor(255, 255, 255, 50), 1))
            p.drawLine(self._hover_x, draw_top, self._hover_x, draw_bottom - 1)
        self._draw_level_meter(p, w, hud_h)
        self._draw_recent_level_graph(p, hud_h)
        self._draw_perf_overlay(p, w, hud_h)
        p.end()

    def _draw_stereo_meter_row(self, p: QPainter, x: int, y: int, w: int, h: int, label: str, level: float, peak_hold: float):
        p.setPen(QPen(QColor(220, 220, 220), 1))
        p.drawText(x, y + h - 1, label)
        meter_x = x + 12
        meter_w = max(40, w - 12)
        p.fillRect(meter_x, y, meter_w, h, QColor(10, 10, 10, 235))
        green_w = int(meter_w * 0.68)
        yellow_w = int(meter_w * 0.18)
        red_w = meter_w - green_w - yellow_w
        fill_w = int(max(0.0, min(1.0, level)) * meter_w)
        if fill_w > 0:
            green_fill = min(fill_w, green_w)
            if green_fill > 0:
                p.fillRect(meter_x, y, green_fill, h, QColor(70, 255, 90, 220))
            yellow_fill = min(max(0, fill_w - green_w), yellow_w)
            if yellow_fill > 0:
                p.fillRect(meter_x + green_w, y, yellow_fill, h, QColor(255, 210, 70, 220))
            red_fill = min(max(0, fill_w - green_w - yellow_w), red_w)
            if red_fill > 0:
                p.fillRect(meter_x + green_w + yellow_w, y, red_fill, h, QColor(255, 80, 80, 220))
        peak_x = meter_x + min(meter_w - 1, max(0, int(max(0.0, min(1.0, peak_hold)) * meter_w)))
        p.setPen(QPen(QColor(90, 170, 255), 2))
        p.drawLine(peak_x, y, peak_x, y + h)
        p.setPen(QPen(QColor(55, 55, 55), 1))
        for divider in (green_w, green_w + yellow_w):
            p.drawLine(meter_x + divider, y, meter_x + divider, y + h)
        p.setPen(QPen(QColor(220, 220, 220), 1))
        p.drawRect(meter_x, y, meter_w, h)

    def _draw_meter_scale(self, p: QPainter, meter_x: int, meter_w: int, y: int):
        marks = [(-36, 0.0), (-24, 0.33), (-12, 0.66), (-6, 0.82), (0, 1.0)]
        p.setPen(QPen(QColor(170, 170, 170), 1))
        for db, ratio in marks:
            x = meter_x + int(ratio * meter_w)
            p.drawLine(x, y + 9, x, y + 11)
            label_x = x - 9 if db < 0 else x - 3
            p.drawText(label_x, y + 8, f"{db}")

    def _draw_level_meter(self, p: QPainter, w: int, hud_h: int):
        meter_x = 10
        meter_w = max(120, w - 250)
        row_h = 8
        gap = 4
        base_y = max(12, (hud_h - ((row_h * 2) + gap)) // 2)
        meter_bar_x = meter_x + 12
        meter_bar_w = max(40, meter_w - 12)
        self._draw_meter_scale(p, meter_bar_x, meter_bar_w, 1)
        self._draw_stereo_meter_row(p, meter_x, base_y, meter_w, row_h, "L", self._display_left_level, self._peak_hold_left)
        self._draw_stereo_meter_row(
            p, meter_x, base_y + row_h + gap, meter_w, row_h, "R", self._display_right_level, self._peak_hold_right
        )

    def _draw_recent_level_graph(self, p: QPainter, hud_h: int):
        box_w = 90
        box_h = max(20, hud_h - 8)
        box_x = 12 + max(120, self.width() - 250) + 10
        box_y = (hud_h - box_h) // 2
        p.fillRect(box_x, box_y, box_w, box_h, QColor(0, 0, 0, 150))
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.drawRect(box_x, box_y, box_w, box_h)
        if self._levels_hz <= 0 or not self._levels.size:
            return
        history_ms = 500
        end_idx = int((self._position_ms / 1000.0) * self._levels_hz)
        span = max(1, int((history_ms / 1000.0) * self._levels_hz))
        start_idx = max(0, end_idx - span + 1)
        window = self._levels[start_idx:end_idx + 1]
        if window.size <= 1:
            return
        p.setPen(QPen(QColor(80, 255, 160), 1))
        last_x = box_x
        last_y = box_y + box_h - int(float(window[0]) * (box_h - 4)) - 2
        for i in range(1, len(window)):
            x = box_x + int((i / max(1, len(window) - 1)) * (box_w - 1))
            y = box_y + box_h - int(float(window[i]) * (box_h - 4)) - 2
            p.drawLine(last_x, last_y, x, y)
            last_x, last_y = x, y

    def _draw_perf_overlay(self, p: QPainter, w: int, hud_h: int):
        hz_label = f"{self._paint_hz:4.1f} Hz" if self._paint_hz > 0 else "--.- Hz"
        bpm_label = f"{self._current_bpm:5.1f} BPM" if self._current_bpm > 0 else "--.- BPM"
        box_h = max(24, hud_h - 6)
        p.fillRect(w - 114, 3, 108, box_h, QColor(0, 0, 0, 140))
        p.setPen(QPen(QColor(220, 220, 220), 1))
        p.drawText(w - 110, 15, hz_label)
        p.drawText(w - 110, min(hud_h - 5, 28), bpm_label)

    def _draw_time_scale(self, p: QPainter, w: int, y_offset: int):
        if self._duration_ms <= 0:
            return
        start_s = self._view_start_ms / 1000.0
        end_s = self._view_end_ms / 1000.0
        visible_s = max(0.001, end_s - start_s)
        ticks_target = max(2, w // 100)
        raw_step = visible_s / ticks_target
        candidates = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
        step = candidates[-1]
        for c in candidates:
            if c >= raw_step:
                step = c
                break
        first = math.ceil(start_s / step) * step
        p.setPen(QPen(QColor(255, 255, 255), 1))
        y = y_offset + 9
        t = first
        while t <= end_s + 0.001:
            x = int(((t - start_s) / visible_s) * w)
            if 0 <= x < w:
                p.drawLine(x, y_offset, x, y_offset + 5)
                p.drawText(x + 3, y, fmt_seconds(int(t)))
            t += step

    def _draw_minimap(self, p: QPainter, w: int, h: int, mini_h: int):
        y = h - mini_h
        p.fillRect(0, y, w, mini_h, QColor(42, 42, 42))
        if self._duration_ms <= 0:
            return
        x0 = int((self._view_start_ms / self._duration_ms) * w)
        x1 = int((self._view_end_ms / self._duration_ms) * w)
        p.fillRect(x0, y, max(2, x1 - x0), mini_h, QColor(100, 100, 100))


class MusicPlayer(QMainWindow):
    MAX_WAVEFORM_PEAKS = 600_000
    DEFAULT_WAVEFORM_WINDOW_MS = 10 * 1000
    DEFAULT_REFRESH_FPS = 60
    REFRESH_FPS_OPTIONS = (30, 60, 120, 144)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Player")
        self.setGeometry(100, 100, 1200, 760)
        self.setMinimumSize(920, 620)

        self.config = self._load_config()
        self.playlist: list[Track] = []
        self.current_track_index = -1
        self.last_folder = self._config_last_folder()
        self.refresh_fps = self._config_refresh_fps()
        self.loader: WaveformSegmentLoader | None = None
        self.waveform_request_id = 0
        self._waveform_request_key_by_id: dict[int, tuple] = {}
        self._waveform_request_meta_by_id: dict[int, dict] = {}
        self._active_waveform_request_key: tuple | None = None
        self._waveform_cache: OrderedDict[tuple, tuple[object, LoudnessMetrics | None, WaveformAnalysis | None]] = OrderedDict()
        self._pending_waveform_request: tuple[int, int, str] | None = None
        self._pending_waveform_force = False
        self._waveform_request_timer = QTimer(self)
        self._waveform_request_timer.setSingleShot(True)
        self._waveform_request_timer.timeout.connect(self._flush_waveform_request)
        self._wave_loaded_for_track = False
        self._current_waveform_target_peaks = 0
        self.track_loudness: dict[str, LoudnessMetrics] = {}
        self.normalize_enabled = True
        self.target_rms = 0.24
        self.target_peak = 0.98
        self.max_normalize_gain = 1.8
        self.slider_drag_preview = False
        self._last_displayed_second = -1
        self._playback_anchor_ms = 0
        self._playback_anchor_ts = 0.0

        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(1.0)
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._on_player_position_changed)
        self.player.durationChanged.connect(self._on_player_duration_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.errorOccurred.connect(self._on_player_error)
        self._playback_refresh_timer = QTimer(self)
        self._playback_refresh_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._playback_refresh_timer.timeout.connect(self._refresh_playback_view)
        self._refresh_rate_actions: dict[int, QAction] = {}
        self._set_refresh_fps(self.refresh_fps, persist=False)

        self._build_ui()
        self._apply_style()
        self._load_tree_root(self.last_folder or os.path.expanduser("~"))

    CFG_PATH = os.path.join(os.path.expanduser("~"), ".music_player_config")

    def _load_config(self) -> dict[str, object]:
        if not os.path.exists(self.CFG_PATH):
            return {}
        try:
            with open(self.CFG_PATH, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                return {}
            if raw.startswith("{"):
                loaded = json.loads(raw)
                return loaded if isinstance(loaded, dict) else {}
            if os.path.isdir(raw):
                return {"last_folder": raw}
        except Exception:
            pass
        return {}

    def _config_last_folder(self) -> str | None:
        value = self.config.get("last_folder")
        if isinstance(value, str) and os.path.isdir(value):
            return value
        return None

    def _config_refresh_fps(self) -> int:
        value = self.config.get("refresh_fps")
        if isinstance(value, int) and value in self.REFRESH_FPS_OPTIONS:
            return value
        return self.DEFAULT_REFRESH_FPS

    def _save_config(self):
        try:
            with open(self.CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f)
        except Exception:
            pass

    def _save_last_folder(self, path: str):
        self.last_folder = path
        self.config["last_folder"] = path
        self._save_config()

    def _set_refresh_fps(self, fps: int, persist: bool = True):
        if fps not in self.REFRESH_FPS_OPTIONS:
            fps = self.DEFAULT_REFRESH_FPS
        self.refresh_fps = fps
        self._playback_refresh_timer.setInterval(max(1, int(round(1000 / fps))))
        for option, action in self._refresh_rate_actions.items():
            action.setChecked(option == fps)
        if persist:
            self.config["refresh_fps"] = fps
            self._save_config()

    def _build_ui(self):
        self._setup_menu()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(8, 8, 4, 8)
        left_l.setSpacing(6)
        left_l.addWidget(QLabel("File Browser"))

        self.fs_model = QFileSystemModel(self)
        self.fs_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)
        self.fs_model.setNameFilters(["*.mp3", "*.wav"])
        self.fs_model.setNameFilterDisables(False)

        self.tree = QTreeView()
        self.tree.setModel(self.fs_model)
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)
        self.tree.doubleClicked.connect(self._on_tree_double_click)
        left_l.addWidget(self.tree, 1)

        add_btn = QPushButton("Add to Playlist")
        add_btn.clicked.connect(self._add_selected_tree_items)
        left_l.addWidget(add_btn)
        splitter.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(4, 8, 8, 8)
        right_l.setSpacing(6)
        right_l.addWidget(QLabel("Playlist"))

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["#", "Title", "Artist", "Year", "Date", "Duration"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        vertical_header = self.table.verticalHeader()
        assert vertical_header is not None
        vertical_header.setVisible(False)
        horizontal_header = self.table.horizontalHeader()
        assert horizontal_header is not None
        for i in range(6):
            horizontal_header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
        horizontal_header.sectionClicked.connect(self._sort_table)
        self.table.itemDoubleClicked.connect(lambda item: self.play_track(item.row()))
        right_l.addWidget(self.table, 1)

        row = QHBoxLayout()
        rem_btn = QPushButton("Remove")
        rem_btn.clicked.connect(self._remove_selected_tracks)
        row.addWidget(rem_btn)
        row.addStretch()
        up_btn = QPushButton("Up")
        up_btn.clicked.connect(self._move_up)
        row.addWidget(up_btn)
        down_btn = QPushButton("Down")
        down_btn.clicked.connect(self._move_down)
        row.addWidget(down_btn)
        right_l.addLayout(row)
        splitter.addWidget(right)
        splitter.setSizes([320, 880])

        playbar = QWidget()
        playbar.setFixedHeight(210)
        pb = QVBoxLayout(playbar)
        pb.setContentsMargins(12, 8, 12, 8)
        pb.setSpacing(6)

        self.track_label = QLabel("No track selected")
        pb.addWidget(self.track_label)

        wr = QHBoxLayout()
        self.current_time_label = QLabel("0:00")
        self.current_time_label.setFixedWidth(60)
        wr.addWidget(self.current_time_label)
        self.waveform = WaveformView()
        self.waveform.seek_preview.connect(self._on_seek_preview)
        self.waveform.seek_commit.connect(self._on_seek_commit)
        self.waveform.zoom_changed.connect(self._on_waveform_zoom_changed)
        self.waveform.viewport_changed.connect(self._on_waveform_viewport_changed)
        wr.addWidget(self.waveform, 1)
        self.total_time_label = QLabel("0:00")
        self.total_time_label.setFixedWidth(60)
        wr.addWidget(self.total_time_label)
        pb.addLayout(wr)

        cr = QHBoxLayout()
        self.prev_btn = QPushButton("<<")
        self.prev_btn.clicked.connect(self.prev_track)
        cr.addWidget(self.prev_btn)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        cr.addWidget(self.play_btn)
        self.next_btn = QPushButton(">>")
        self.next_btn.clicked.connect(self.next_track)
        cr.addWidget(self.next_btn)
        cr.addSpacing(16)

        cr.addWidget(QLabel("Volume"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(120)
        self.volume_slider.valueChanged.connect(self._apply_volume)
        cr.addWidget(self.volume_slider)

        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setChecked(True)
        self.normalize_check.stateChanged.connect(self._on_normalize_toggled)
        cr.addWidget(self.normalize_check)
        cr.addSpacing(16)

        cr.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 100)
        self.zoom_slider.setValue(1)
        self.zoom_slider.setFixedWidth(160)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        cr.addWidget(self.zoom_slider)
        cr.addStretch()
        pb.addLayout(cr)
        root.addWidget(playbar)

        self.status_label = QLabel("")
        pb.addWidget(self.status_label)

        self.waveform.set_default_window_limit_ms(self.DEFAULT_WAVEFORM_WINDOW_MS)
        self._refresh_table()

    def _setup_menu(self):
        mb = self.menuBar()
        assert mb is not None
        file_menu = mb.addMenu("File")
        assert file_menu is not None
        open_action = QAction("Open Folder", self)
        open_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        playlist_menu = mb.addMenu("Playlist")
        assert playlist_menu is not None
        save_action = QAction("Save Playlist", self)
        save_action.triggered.connect(self._save_playlist)
        playlist_menu.addAction(save_action)
        load_action = QAction("Load Playlist", self)
        load_action.triggered.connect(self._load_playlist)
        playlist_menu.addAction(load_action)
        clear_action = QAction("Clear Playlist", self)
        clear_action.triggered.connect(self._clear_playlist)
        playlist_menu.addAction(clear_action)

        settings_menu = mb.addMenu("Settings")
        assert settings_menu is not None
        refresh_menu = settings_menu.addMenu("Waveform Refresh Rate")
        assert refresh_menu is not None
        refresh_group = QActionGroup(self)
        refresh_group.setExclusive(True)
        self._refresh_rate_actions.clear()
        for fps in self.REFRESH_FPS_OPTIONS:
            action = QAction(f"{fps} FPS", self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=fps: self._set_refresh_fps(value))
            refresh_group.addAction(action)
            refresh_menu.addAction(action)
            self._refresh_rate_actions[fps] = action
        self._set_refresh_fps(self.refresh_fps, persist=False)

        help_menu = mb.addMenu("Help")
        assert help_menu is not None
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #1d1d1d; color: #e2e2e2; }
            QTreeView, QTableWidget { background: #242424; border: none; }
            QHeaderView::section { background: #1a1a1a; color: #9a9a9a; border: none; padding: 6px; }
            QPushButton { background: #333; border: none; border-radius: 4px; padding: 6px 10px; }
            QPushButton:hover { background: #3d3d3d; }
            QLabel { color: #d9d9d9; }
            QSlider::groove:horizontal { background: #3a3a3a; height: 5px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #1db954; width: 12px; margin: -4px 0; border-radius: 6px; }
            QCheckBox { color: #d9d9d9; }
            """
        )

    def _load_tree_root(self, path: str):
        idx = self.fs_model.setRootPath(path)
        self.tree.setRootIndex(idx)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._save_last_folder(folder)
            self._load_tree_root(folder)

    def _on_tree_double_click(self, idx):
        path = self.fs_model.filePath(idx)
        if os.path.isdir(path):
            return
        if path.lower().endswith((".mp3", ".wav")):
            self._add_track(path)

    def _add_selected_tree_items(self):
        selection_model = self.tree.selectionModel()
        assert selection_model is not None
        rows = selection_model.selectedRows()
        for idx in rows:
            path = self.fs_model.filePath(idx)
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for f in sorted(files):
                        if f.lower().endswith((".mp3", ".wav")):
                            self._add_track(os.path.join(root, f))
            elif path.lower().endswith((".mp3", ".wav")):
                self._add_track(path)

    def _add_track(self, filepath: str):
        title = os.path.splitext(os.path.basename(filepath))[0]
        artist = "Unknown Artist"
        year = ""
        duration_s = 0
        try:
            meta = MP3(filepath)
            tags = meta.tags or {}
            if "TIT2" in tags and tags["TIT2"].text:
                title = str(tags["TIT2"].text[0])
            if "TPE1" in tags and tags["TPE1"].text:
                artist = str(tags["TPE1"].text[0])
            if "TDRC" in tags and tags["TDRC"].text:
                year = str(tags["TDRC"].text[0])
            elif "TYER" in tags and tags["TYER"].text:
                year = str(tags["TYER"].text[0])
            duration_s = int(meta.info.length)
        except (MutagenError, AttributeError, Exception):
            pass

        mtime = os.path.getmtime(filepath)
        date = datetime.datetime.fromtimestamp(mtime).strftime("%d.%m.%Y")
        self.playlist.append(Track(filepath, title, artist, year, date, duration_s, mtime))
        self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self.playlist))
        for i, t in enumerate(self.playlist):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(t.title))
            self.table.setItem(i, 2, QTableWidgetItem(t.artist))
            self.table.setItem(i, 3, QTableWidgetItem(t.year))
            self.table.setItem(i, 4, QTableWidgetItem(t.date))
            self.table.setItem(i, 5, QTableWidgetItem(fmt_seconds(t.duration_s)))
        self._resize_columns()

    def _resize_columns(self):
        w = self.table.width()
        widths = [
            int(w * 0.05),
            int(w * 0.33),
            int(w * 0.30),
            int(w * 0.08),
            int(w * 0.12),
        ]
        widths.append(max(50, w - sum(widths)))
        for i, v in enumerate(widths):
            self.table.setColumnWidth(i, v)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self._resize_columns()

    def _sort_table(self, col: int):
        if col == 0:
            return
        reverse = getattr(self, "_sort_reverse", False) if getattr(self, "_sort_col", -1) == col else False
        self._sort_col = col
        self._sort_reverse = not reverse
        if col == 1:
            self.playlist.sort(key=lambda t: t.title.lower(), reverse=self._sort_reverse)
        elif col == 2:
            self.playlist.sort(key=lambda t: t.artist.lower(), reverse=self._sort_reverse)
        elif col == 3:
            self.playlist.sort(key=lambda t: t.year, reverse=self._sort_reverse)
        elif col == 4:
            self.playlist.sort(key=lambda t: t.mtime, reverse=self._sort_reverse)
        elif col == 5:
            self.playlist.sort(key=lambda t: t.duration_s, reverse=self._sort_reverse)
        self._refresh_table()

    def _remove_selected_tracks(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.playlist):
                self.playlist.pop(r)
        if self.current_track_index >= len(self.playlist):
            self.current_track_index = len(self.playlist) - 1
        self._refresh_table()

    def _move_up(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        for r in rows:
            if r > 0:
                self.playlist[r], self.playlist[r - 1] = self.playlist[r - 1], self.playlist[r]
        self._refresh_table()

    def _move_down(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            if r < len(self.playlist) - 1:
                self.playlist[r], self.playlist[r + 1] = self.playlist[r + 1], self.playlist[r]
        self._refresh_table()

    def _save_playlist(self):
        if not self.playlist:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Playlist", "", "M3U Playlist (*.m3u)")
        if not filename:
            return
        with open(filename, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            for t in self.playlist:
                f.write(f"#EXTINF:{t.duration_s},{t.artist} - {t.title}\n")
                f.write(t.filepath + "\n")
        QMessageBox.information(self, "Saved", "Playlist saved.")

    def _load_playlist(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Playlist", "", "M3U Playlist (*.m3u)")
        if not filename:
            return
        self.playlist.clear()
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and os.path.exists(line):
                    self._add_track(line)
        QMessageBox.information(self, "Loaded", "Playlist loaded.")

    def _clear_playlist(self):
        self._stop_loader()
        self.player.stop()
        self.playlist.clear()
        self.current_track_index = -1
        self.track_label.setText("No track selected")
        self.current_time_label.setText("0:00")
        self.total_time_label.setText("0:00")
        self.waveform.set_track_duration(0)
        self.waveform.set_loading()
        self.waveform.set_error("No waveform")
        self.zoom_slider.setValue(1)
        self._wave_loaded_for_track = False
        self._refresh_table()

    def play_track(self, index: int):
        if not (0 <= index < len(self.playlist)):
            return
        self._waveform_request_timer.stop()
        self._pending_waveform_request = None
        self._pending_waveform_force = False
        self._stop_loader()
        self.waveform_request_id += 1
        self.current_track_index = index
        self._wave_loaded_for_track = False
        self._current_waveform_target_peaks = 0
        t = self.playlist[index]
        self.track_label.setText(f"{t.title} - {t.artist}")
        self.current_time_label.setText("0:00")
        self._last_displayed_second = 0
        self.total_time_label.setText(fmt_seconds(t.duration_s))
        self.table.selectRow(index)
        self.waveform.set_default_window_limit_ms(self.DEFAULT_WAVEFORM_WINDOW_MS)
        self.waveform.set_track_duration(t.duration_s * 1000)
        self.waveform.set_follow_playback(False)
        self.waveform.set_loading()
        self.waveform.set_position(0)
        self.zoom_slider.setValue(1)
        self.status_label.setText("")
        self._set_playback_anchor(0)
        self._schedule_full_waveform_request(force=True, immediate=True)

        self.player.setSource(QUrl.fromLocalFile(t.filepath))
        self.player.play()
        self._apply_volume()

    def _stop_loader(self):
        self._active_waveform_request_key = None
        if self.loader is None:
            return
        self.loader.cancel()
        try:
            self.loader.segment_partial.disconnect()
            self.loader.segment_final.disconnect()
            self.loader.analysis_ready.disconnect()
            self.loader.loudness_ready.disconnect()
            self.loader.error.disconnect()
        except Exception:
            pass
        self.loader = None

    def _schedule_waveform_request(
        self, start_ms: int, end_ms: int, force: bool = False, immediate: bool = False, mode: str = "full"
    ):
        self._pending_waveform_request = (start_ms, end_ms, mode)
        self._pending_waveform_force = self._pending_waveform_force or force
        if immediate:
            self._waveform_request_timer.start(0)
        else:
            self._waveform_request_timer.start(8)

    def _flush_waveform_request(self):
        if self._pending_waveform_request is None:
            return
        start_ms, end_ms, mode = self._pending_waveform_request
        force = self._pending_waveform_force
        self._pending_waveform_request = None
        self._pending_waveform_force = False
        self._request_waveform_segment(start_ms, end_ms, force=force, mode=mode)

    def _schedule_full_waveform_request(self, force: bool = False, immediate: bool = False):
        if self.current_track_index < 0:
            return
        duration_ms = max(1, self.playlist[self.current_track_index].duration_s * 1000)
        self._schedule_waveform_request(0, duration_ms, force=force, immediate=immediate)

    def _adaptive_target_peaks(self, duration_ms: int) -> int:
        width_px = max(1, self.waveform.width())
        view_start, view_end = self.waveform.get_viewport_ms()
        visible_span_ms = max(1, view_end - view_start)
        scaled = math.ceil((width_px * max(1, duration_ms)) / visible_span_ms)
        return max(width_px, min(self.MAX_WAVEFORM_PEAKS, scaled))

    def _request_waveform_segment(self, start_ms: int, end_ms: int, force: bool = False, mode: str = "full"):
        if self.current_track_index < 0:
            return
        track = self.playlist[self.current_track_index]
        duration_ms = max(1, track.duration_s * 1000)
        start_ms = max(0, min(duration_ms - 1, start_ms))
        end_ms = max(start_ms + 1, min(duration_ms, end_ms))
        target_peaks = self._adaptive_target_peaks(duration_ms)
        granularity_ms = max(10, int((end_ms - start_ms) / max(1, target_peaks)))
        q_start = (start_ms // granularity_ms) * granularity_ms
        q_end = max(q_start + 1, ((end_ms + granularity_ms - 1) // granularity_ms) * granularity_ms)
        q_end = min(duration_ms, q_end)
        key = (track.filepath, q_start, q_end, target_peaks)

        if not force and key in self._waveform_cache:
            peaks, loudness, analysis = self._waveform_cache[key]
            self._waveform_cache.move_to_end(key)
            self.waveform.set_final(q_start, q_end, peaks)
            self.waveform.set_analysis_data(analysis)
            self._current_waveform_target_peaks = target_peaks
            if loudness is not None:
                self._on_loudness_ready(track.filepath, loudness["rms"], loudness["peak"])
            self._wave_loaded_for_track = True
            return

        if self.loader is not None and self._active_waveform_request_key == key:
            return

        self._stop_loader()
        self.waveform_request_id += 1
        request_id = self.waveform_request_id
        self._waveform_request_key_by_id[request_id] = key
        self._waveform_request_meta_by_id[request_id] = {
            "mode": mode,
            "view_start": start_ms,
            "view_end": end_ms,
            "q_start": q_start,
            "q_end": q_end,
            "target_peaks": target_peaks,
        }
        self._active_waveform_request_key = key
        self.waveform.set_loading()

        self.loader = WaveformSegmentLoader(
            request_id=request_id,
            filepath=track.filepath,
            duration_ms=duration_ms,
            start_ms=q_start,
            end_ms=q_end,
            target_peaks=target_peaks,
            parent=self,
        )
        self.loader.segment_partial.connect(self._on_segment_partial)
        self.loader.segment_final.connect(self._on_segment_final)
        self.loader.analysis_ready.connect(self._on_segment_analysis)
        self.loader.loudness_ready.connect(self._on_segment_loudness)
        self.loader.error.connect(self._on_segment_error)
        self.loader.start()

    @staticmethod
    def _merge_stereo_segments(base_seg, add_seg):
        if base_seg is None:
            return add_seg
        bs, be, (bl, br) = base_seg
        ns, ne, (nl, nr) = add_seg
        if ne <= bs:
            return ns, be, (np.concatenate((nl, bl)), np.concatenate((nr, br)))
        if ns >= be:
            return bs, ne, (np.concatenate((bl, nl)), np.concatenate((br, nr)))
        # overlap merge (simple trim by time proportion)
        if ns < bs:
            overlap = max(0, ne - bs)
            drop = int((overlap / max(1, ne - ns)) * len(nl))
            nl = nl[: max(0, len(nl) - drop)]
            nr = nr[: max(0, len(nr) - drop)]
            return ns, be, (np.concatenate((nl, bl)), np.concatenate((nr, br)))
        overlap = max(0, be - ns)
        drop = int((overlap / max(1, ne - ns)) * len(nl))
        nl = nl[min(len(nl), drop):]
        nr = nr[min(len(nr), drop):]
        return bs, ne, (np.concatenate((bl, nl)), np.concatenate((br, nr)))

    @staticmethod
    def _compose_view_from_segment(seg, view_start: int, view_end: int, target_peaks: int):
        seg_start, seg_end, (left, right) = seg
        out_l = np.zeros(max(1, target_peaks), dtype=np.float32)
        out_r = np.zeros(max(1, target_peaks), dtype=np.float32)
        seg_span = max(1, seg_end - seg_start)
        view_span = max(1, view_end - view_start)
        for i in range(len(out_l)):
            t = view_start + int((i / max(1, len(out_l) - 1)) * view_span)
            if t < seg_start or t > seg_end:
                continue
            ratio = (t - seg_start) / seg_span
            idx = int(max(0.0, min(1.0, ratio)) * (len(left) - 1))
            out_l[i] = float(left[idx])
            out_r[i] = float(right[idx])
        return out_l, out_r

    def toggle_play(self):
        if not self.playlist:
            return
        if self.current_track_index < 0:
            row = self.table.currentRow()
            self.play_track(row if row >= 0 else 0)
            return
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def prev_track(self):
        if self.current_track_index > 0:
            self.play_track(self.current_track_index - 1)

    def next_track(self):
        if self.current_track_index < len(self.playlist) - 1:
            self.play_track(self.current_track_index + 1)
        elif self.playlist:
            self.play_track(0)

    def _set_playback_anchor(self, ms: int):
        self._playback_anchor_ms = max(0, ms)
        self._playback_anchor_ts = time.perf_counter()

    def _estimated_playback_position(self) -> int:
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            return max(0, self.player.position())
        elapsed_ms = int(max(0.0, time.perf_counter() - self._playback_anchor_ts) * 1000.0)
        estimate = self._playback_anchor_ms + elapsed_ms
        duration_ms = self.player.duration()
        if duration_ms > 0:
            estimate = min(duration_ms, estimate)
        return max(0, estimate)

    def _sync_playback_ui(self, ms: int):
        if self.slider_drag_preview:
            return
        second = max(0, ms // 1000)
        if second != self._last_displayed_second:
            self.current_time_label.setText(fmt_seconds(second))
            self._last_displayed_second = second
        self.waveform.set_position(ms)

    def _refresh_playback_view(self):
        if self.current_track_index < 0:
            return
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            return
        self._sync_playback_ui(self._estimated_playback_position())

    def _on_player_position_changed(self, ms: int):
        self._set_playback_anchor(ms)
        if not self._playback_refresh_timer.isActive():
            self._sync_playback_ui(ms)

    def _on_player_duration_changed(self, ms: int):
        if ms > 0 and self.current_track_index >= 0:
            self.total_time_label.setText(fmt_seconds(ms // 1000))
            if self.playlist[self.current_track_index].duration_s <= 0:
                self.playlist[self.current_track_index].duration_s = int(ms // 1000)
            start, end = self.waveform.get_viewport_ms()
            if end <= start:
                self.waveform.set_track_duration(ms)
            self._schedule_full_waveform_request(force=not self._wave_loaded_for_track, immediate=True)

    def _on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.next_track()

    def _on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setText("Pause")
            self.waveform.set_follow_playback(True)
            self._set_playback_anchor(self.player.position())
            self._playback_refresh_timer.start()
            self._sync_playback_ui(self._estimated_playback_position())
        else:
            self.play_btn.setText("Play")
            self.waveform.set_follow_playback(False)
            self._playback_refresh_timer.stop()
            self._set_playback_anchor(self.player.position())
            self._sync_playback_ui(self.player.position())

    def _on_player_error(self, error, error_string):
        if error_string:
            self.status_label.setText(f"Playback error: {error_string}")

    def _on_seek_preview(self, ms: int):
        self.slider_drag_preview = True
        self.current_time_label.setText(fmt_seconds(ms // 1000))
        self._last_displayed_second = max(0, ms // 1000)

    def _on_seek_commit(self, ms: int):
        self._set_playback_anchor(ms)
        self.player.setPosition(ms)
        self.slider_drag_preview = False
        self._sync_playback_ui(ms)

    def _on_zoom_slider_changed(self, value: int):
        self.waveform.set_zoom_level(value)

    def _on_waveform_zoom_changed(self, value: int):
        if self.zoom_slider.value() == value:
            return
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(value)
        self.zoom_slider.blockSignals(False)

    def _on_waveform_viewport_changed(self, start_ms: int, end_ms: int):
        if self.current_track_index < 0:
            return
        duration_ms = max(1, self.playlist[self.current_track_index].duration_s * 1000)
        desired_target_peaks = self._adaptive_target_peaks(duration_ms)
        seg = self.waveform.get_segment_data()

        if seg is not None:
            seg_start, seg_end, _ = seg
            if seg_start <= 0 and seg_end >= duration_ms and self._current_waveform_target_peaks == desired_target_peaks:
                return

        self._schedule_full_waveform_request(force=not self._wave_loaded_for_track, immediate=False)

    def _on_segment_partial(self, start_ms: int, end_ms: int, peaks, request_id: int):
        # Suppress progressive rendering - waveform will appear all at once when final arrives
        if request_id != self.waveform_request_id:
            return
        # Don't call set_partial - wait for final to appear all at once

    def _on_segment_final(self, start_ms: int, end_ms: int, peaks, request_id: int):
        if request_id != self.waveform_request_id:
            return
        meta = self._waveform_request_meta_by_id.get(request_id, {})
        mode = meta.get("mode", "full")
        key = self._waveform_request_key_by_id.get(request_id)
        if key is not None and peaks is not None:
            loudness_cached = self._waveform_cache[key][1] if key in self._waveform_cache else None
            analysis_cached = self._waveform_cache[key][2] if key in self._waveform_cache else None
            self._waveform_cache[key] = (peaks, loudness_cached, analysis_cached)
            self._waveform_cache.move_to_end(key)
            while len(self._waveform_cache) > 8:
                self._waveform_cache.popitem(last=False)

        if peaks is None:
            self.waveform.set_final(start_ms, end_ms, peaks)
        elif mode == "extend_left" or mode == "extend_right":
            base = self.waveform.get_segment_data()
            merged = self._merge_stereo_segments(base, (start_ms, end_ms, peaks))
            self.waveform.set_final(merged[0], merged[1], merged[2])
        else:
            self.waveform.set_final(start_ms, end_ms, peaks)

        if peaks is not None:
            self._wave_loaded_for_track = True
            self._current_waveform_target_peaks = int(meta.get("target_peaks", 0))
        self._waveform_request_meta_by_id.pop(request_id, None)
        self._waveform_request_key_by_id.pop(request_id, None)

    def _on_segment_analysis(self, analysis: WaveformAnalysis, request_id: int):
        if request_id != self.waveform_request_id:
            return
        key = self._waveform_request_key_by_id.get(request_id)
        if key is not None:
            peaks_cached = self._waveform_cache[key][0] if key in self._waveform_cache else None
            loudness_cached = self._waveform_cache[key][1] if key in self._waveform_cache else None
            self._waveform_cache[key] = (peaks_cached, loudness_cached, analysis)
        self.waveform.set_analysis_data(analysis)

    def _on_segment_loudness(self, rms: float, peak: float, request_id: int):
        if request_id != self.waveform_request_id:
            return
        if self.current_track_index < 0:
            return
        fp = self.playlist[self.current_track_index].filepath
        key = self._waveform_request_key_by_id.get(request_id)
        if key is not None and key in self._waveform_cache:
            peaks, _, analysis = self._waveform_cache[key]
            self._waveform_cache[key] = (peaks, {"rms": rms, "peak": peak}, analysis)
        self._on_loudness_ready(fp, rms, peak)

    def _on_segment_error(self, msg: str, request_id: int):
        if request_id != self.waveform_request_id:
            return
        self.status_label.setText(msg)
        self.waveform.set_error(msg)

    def _on_loudness_ready(self, filepath: str, rms: float, peak: float):
        self.track_loudness[filepath] = {"rms": rms, "peak": peak}
        self._apply_volume()

    def _on_normalize_toggled(self, state: int):
        self.normalize_enabled = bool(state)
        self._apply_volume()

    def _apply_volume(self):
        base = self.volume_slider.value() / 100.0
        if not self.normalize_enabled or self.current_track_index < 0:
            self.audio_output.setVolume(base)
            return
        fp = self.playlist[self.current_track_index].filepath
        loudness = self.track_loudness.get(fp)
        if not loudness:
            self.audio_output.setVolume(base)
            return
        rms = loudness["rms"]
        peak = loudness["peak"]
        if rms <= 0 or peak <= 0:
            self.audio_output.setVolume(base)
            return
        desired_gain = max(1.0, self.target_rms / rms)
        peak_limited_gain = max(1.0, self.target_peak / peak)
        gain = min(self.max_normalize_gain, desired_gain, peak_limited_gain)
        self.audio_output.setVolume(max(0.0, min(1.0, base * gain)))

    def _show_about(self):
        QMessageBox.about(
            self,
            "About",
            "Music Player\n\n"
            "New Qt-based engine for ZorinOS/Linux.\n"
            "Features:\n"
            "- Full waveform rendered in one pass\n"
            "- Centered playhead follow mode\n"
            "- Live loudness and BPM overlays\n"
            "- Visible timeline scale\n"
            "- Mouse wheel and slider zoom\n"
            "- Right-drag panning\n"
            "- Normalize toggle\n"
            "- Playlist save/load (.m3u)\n",
        )

    def closeEvent(self, a0):
        self._playback_refresh_timer.stop()
        self._stop_loader()
        self.player.stop()
        super().closeEvent(a0)


def main():
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    w = MusicPlayer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
