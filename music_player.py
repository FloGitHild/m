#!/usr/bin/env python3
import os
import sys
import math
import time
import wave
import datetime
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from mutagen.mp3 import MP3
from mutagen import MutagenError

from PyQt6.QtCore import Qt, QDir, QUrl, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QColor, QPainter, QPen, QFileSystemModel
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
    HAS_MINIAUDIO = False


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
    rms_ready = pyqtSignal(float, int)
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

    def _emit_from_samples(self, left_samples: np.ndarray, right_samples: np.ndarray):
        if left_samples.size == 0 or right_samples.size == 0:
            self.segment_final.emit(self.start_ms, self.end_ms, None, self.request_id)
            return
        left_peaks = self._to_peaks(left_samples, self.target_peaks)
        right_peaks = self._to_peaks(right_samples, self.target_peaks)
        peaks = (left_peaks, right_peaks)
        self.segment_partial.emit(self.start_ms, self.end_ms, peaks, self.request_id)
        self.segment_final.emit(self.start_ms, self.end_ms, peaks, self.request_id)

        mono = 0.5 * (left_samples + right_samples)
        rms = float(np.sqrt(np.mean(mono * mono)))
        abs_max = float(np.max(np.abs(mono)))
        if abs_max > 0:
            rms /= abs_max
        self.rms_ready.emit(rms, self.request_id)

    def _run_miniaudio_segment(self):
        try:
            window_sec = max(0.001, (self.end_ms - self.start_ms) / 1000.0)
            decode_rate = self._decode_rate_for_window(window_sec, self.target_peaks)
            start_frame = int((self.start_ms / 1000.0) * decode_rate)
            total_frames = int(window_sec * decode_rate)
            read_frames = 0
            chunk_frames = max(2048, decode_rate // 2)
            collected_left: list[np.ndarray] = []
            collected_right: list[np.ndarray] = []

            stream = miniaudio.stream_file(
                self.filepath,
                output_format=miniaudio.SampleFormat.SIGNED16,
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

                if len(collected_left) % 3 == 0:
                    tmp_left = np.concatenate(collected_left)
                    tmp_right = np.concatenate(collected_right)
                    peaks = (
                        self._to_peaks(tmp_left, self.target_peaks),
                        self._to_peaks(tmp_right, self.target_peaks),
                    )
                    self.segment_partial.emit(self.start_ms, self.end_ms, peaks, self.request_id)

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

        self.setMinimumHeight(100)
        self.setMouseTracking(True)

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
        if peaks is not None:
            self._status = ""
        self.update()

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
        self.zoom_changed.emit(self._zoom_level)
        self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
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
        self.update()

    def clear_for_view(self, start_ms: int, end_ms: int):
        self._segment_start_ms = start_ms
        self._segment_end_ms = end_ms
        self._peaks = None
        self._loading = True
        self._status = "Loading waveform ..."
        self.update()

    def set_error(self, msg: str):
        self._loading = False
        if self._peaks is None:
            self._status = msg
        self.update()

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
        if peaks is None:
            self._status = "Waveform unavailable"
        else:
            self._status = ""
        self.update()

    def set_position(self, ms: int):
        self._position_ms = max(0, ms)
        if not self._drag_seek:
            if self._auto_scroll() and not self._follow_playback:
                self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
        self.update()

    def set_zoom_level(self, level: int):
        level = max(1, min(100, int(level)))
        self._zoom_level = level
        self._apply_zoom_around_center()
        self.zoom_changed.emit(level)
        self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
        self.update()

    def reset_zoom(self):
        self._zoom_level = 1
        self._set_default_initial_view()
        self.zoom_changed.emit(self._zoom_level)
        self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
        self.update()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.set_zoom_level(self._zoom_level + 2)
        else:
            self.set_zoom_level(self._zoom_level - 2)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.reset_zoom()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._drag_pan = True
            self._pan_start_x = int(event.position().x())
            self._pan_start_range = (self._view_start_ms, self._view_end_ms)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_seek = True
            self._edge_anchor_x = int(event.position().x())
            self._update_edge_scroll_state(self._edge_anchor_x)
            ms = self._x_to_ms(event.position().x())
            self.seek_preview.emit(ms)
            self.update()

    def mouseMoveEvent(self, event):
        x = int(event.position().x())
        self._hover_x = x
        if self._drag_seek:
            self._edge_anchor_x = x
            self._update_edge_scroll_state(x)
            ms = self._x_to_ms(x)
            self.seek_preview.emit(ms)
            self._position_ms = ms
        elif self._drag_pan and self._duration_ms > 0:
            dx = int(event.position().x()) - self._pan_start_x
            span = max(1, self._pan_start_range[1] - self._pan_start_range[0])
            ms_per_px = span / max(1, self.width())
            shift = int(dx * ms_per_px)
            start = self._pan_start_range[0] - shift
            end = self._pan_start_range[1] - shift
            self._set_view_window(start, end)
            self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)
            self._update_edge_scroll_state(-1)
        else:
            self._update_edge_scroll_state(-1)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self._drag_pan:
            self._drag_pan = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_seek:
            self._drag_seek = False
            self._update_edge_scroll_state(-1)
            ms = self._position_ms
            self.seek_commit.emit(ms)
            self.update()

    def leaveEvent(self, event):
        self._hover_x = -1
        if not self._drag_seek:
            self._update_edge_scroll_state(-1)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._duration_ms > 0:
            self.viewport_changed.emit(self._view_start_ms, self._view_end_ms)

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
        span = max(1, end - start)
        max_start = max(0, self._duration_ms - span)
        start = max(0, min(max_start, start))
        end = start + span
        self._view_start_ms = start
        self._view_end_ms = min(self._duration_ms, end)

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
            start = center - span // 2
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
            self.update()
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

    def paintEvent(self, event):
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
        mini_h = 8
        draw_h = h - mini_h
        center = draw_h // 2

        p.fillRect(0, 0, w, h, QColor(16, 16, 16))
        p.setPen(QPen(QColor(42, 42, 42), 1))
        p.drawLine(0, center, w, center)

        if self._peaks is not None and self._duration_ms > 0:
            left_peaks, right_peaks = self._peaks
            n = min(len(left_peaks), len(right_peaks))
            if n <= 0:
                left_peaks = np.empty(0, dtype=np.float32)
                right_peaks = np.empty(0, dtype=np.float32)
                n = 0
            blue = QPen(QColor(40, 120, 255, 110), 1)
            yellow = QPen(QColor(255, 220, 70, 110), 1)
            green = QPen(QColor(80, 230, 90, 150), 1)
            seg_span = max(1, self._segment_end_ms - self._segment_start_ms)
            view_span = max(1, self._view_end_ms - self._view_start_ms)
            for x in range(w):
                if n <= 0:
                    break
                t = self._view_start_ms + int((x / max(1, w - 1)) * view_span)
                if t < self._segment_start_ms or t > self._segment_end_ms:
                    continue
                r = (t - self._segment_start_ms) / seg_span
                idx = int(max(0.0, min(1.0, r)) * (n - 1))
                lv = float(left_peaks[idx])
                rv = float(right_peaks[idx])
                lamp = max(1, int(lv * (center - 3)))
                ramp = max(1, int(rv * (center - 3)))
                oamp = min(lamp, ramp)
                p.setPen(blue)
                p.drawLine(x, center - lamp, x, center + lamp)
                p.setPen(yellow)
                p.drawLine(x, center - ramp, x, center + ramp)
                p.setPen(green)
                p.drawLine(x, center - oamp, x, center + oamp)
        else:
            msg = self._status
            if msg:
                p.setPen(QPen(QColor(120, 120, 120), 1))
                p.drawText(0, 0, w, draw_h, Qt.AlignmentFlag.AlignCenter, msg)

        self._draw_time_scale(p, w)

        if self._duration_ms > 0:
            pos_x = self._ms_to_x(self._position_ms)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(pos_x, 0, pos_x, draw_h - 1)

        if self._hover_x >= 0 and not self._drag_pan and not self._drag_seek:
            p.setPen(QPen(QColor(255, 255, 255, 50), 1))
            p.drawLine(self._hover_x, 0, self._hover_x, draw_h - 1)

        self._draw_perf_overlay(p, w)
        self._draw_minimap(p, w, h, mini_h)
        p.end()

    def _draw_perf_overlay(self, p: QPainter, w: int):
        label = f"{self._paint_hz:4.1f} Hz" if self._paint_hz > 0 else "--.- Hz"
        p.fillRect(w - 78, 4, 72, 16, QColor(0, 0, 0, 140))
        p.setPen(QPen(QColor(220, 220, 220), 1))
        p.drawText(w - 74, 16, label)

    def _draw_time_scale(self, p: QPainter, w: int):
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
        p.setPen(QPen(QColor(95, 95, 95), 1))
        y = 9
        t = first
        while t <= end_s + 0.001:
            x = int(((t - start_s) / visible_s) * w)
            if 0 <= x < w:
                p.drawLine(x, 0, x, 5)
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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Player")
        self.setGeometry(100, 100, 1200, 760)
        self.setMinimumSize(920, 620)

        self.playlist: list[Track] = []
        self.current_track_index = -1
        self.last_folder = self._load_last_folder()
        self.loader: WaveformSegmentLoader | None = None
        self.waveform_request_id = 0
        self._waveform_request_key_by_id: dict[int, tuple] = {}
        self._waveform_request_meta_by_id: dict[int, dict] = {}
        self._active_waveform_request_key: tuple | None = None
        self._waveform_cache: OrderedDict[tuple, tuple[object, float | None]] = OrderedDict()
        self._pending_waveform_request: tuple[int, int, str] | None = None
        self._pending_waveform_force = False
        self._waveform_request_timer = QTimer(self)
        self._waveform_request_timer.setSingleShot(True)
        self._waveform_request_timer.timeout.connect(self._flush_waveform_request)
        self.dynamic_waveform_enabled = True
        self.default_10sec_view_enabled = True
        self._wave_loaded_for_track = False
        self._current_waveform_target_peaks = 0
        self.track_rms: dict[str, float] = {}
        self.normalize_enabled = True
        self.target_rms = 0.20
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
        self._playback_refresh_timer.setInterval(16)
        self._playback_refresh_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._playback_refresh_timer.timeout.connect(self._refresh_playback_view)

        self._build_ui()
        self._apply_style()
        self._load_tree_root(self.last_folder or os.path.expanduser("~"))

    CFG_PATH = os.path.join(os.path.expanduser("~"), ".music_player_config")

    def _load_last_folder(self) -> str | None:
        try:
            if os.path.exists(self.CFG_PATH):
                value = open(self.CFG_PATH, "r", encoding="utf-8").read().strip()
                if os.path.isdir(value):
                    return value
        except Exception:
            pass
        return None

    def _save_last_folder(self, path: str):
        try:
            with open(self.CFG_PATH, "w", encoding="utf-8") as f:
                f.write(path)
        except Exception:
            pass

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
        self.table.verticalHeader().setVisible(False)
        for i in range(6):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().sectionClicked.connect(self._sort_table)
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
        self.default_view_check = QCheckBox("10s Default View")
        self.default_view_check.setChecked(True)
        self.default_view_check.stateChanged.connect(self._on_default_view_toggled)
        cr.addWidget(self.default_view_check)
        self.dynamic_check = QCheckBox("Dynamic Waveform")
        self.dynamic_check.setChecked(True)
        self.dynamic_check.stateChanged.connect(self._on_dynamic_waveform_toggled)
        cr.addWidget(self.dynamic_check)
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

        self.waveform.set_default_window_limit_ms(10 * 1000)
        self._refresh_table()

    def _setup_menu(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("File")
        open_action = QAction("Open Folder", self)
        open_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        playlist_menu = mb.addMenu("Playlist")
        save_action = QAction("Save Playlist", self)
        save_action.triggered.connect(self._save_playlist)
        playlist_menu.addAction(save_action)
        load_action = QAction("Load Playlist", self)
        load_action.triggered.connect(self._load_playlist)
        playlist_menu.addAction(load_action)
        clear_action = QAction("Clear Playlist", self)
        clear_action.triggered.connect(self._clear_playlist)
        playlist_menu.addAction(clear_action)

        help_menu = mb.addMenu("Help")
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
        rows = self.tree.selectionModel().selectedRows()
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
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
        self.waveform.set_default_window_limit_ms(10 * 1000 if self.default_10sec_view_enabled else 0)
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
            self.loader.rms_ready.disconnect()
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
            peaks, rms = self._waveform_cache[key]
            self._waveform_cache.move_to_end(key)
            self.waveform.set_final(q_start, q_end, peaks)
            self._current_waveform_target_peaks = target_peaks
            if rms is not None:
                self._on_rms_ready(track.filepath, rms)
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
        self.loader.rms_ready.connect(self._on_segment_rms)
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
            rms_cached = self._waveform_cache[key][1] if key in self._waveform_cache else None
            self._waveform_cache[key] = (peaks, rms_cached)
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

    def _on_segment_rms(self, rms: float, request_id: int):
        if request_id != self.waveform_request_id:
            return
        if self.current_track_index < 0:
            return
        fp = self.playlist[self.current_track_index].filepath
        key = self._waveform_request_key_by_id.get(request_id)
        if key is not None and key in self._waveform_cache:
            peaks, _ = self._waveform_cache[key]
            self._waveform_cache[key] = (peaks, rms)
        self._on_rms_ready(fp, rms)

    def _on_segment_error(self, msg: str, request_id: int):
        if request_id != self.waveform_request_id:
            return
        self.status_label.setText(msg)
        self.waveform.set_error(msg)

    def _on_rms_ready(self, filepath: str, rms: float):
        self.track_rms[filepath] = rms
        self._apply_volume()

    def _on_normalize_toggled(self, state: int):
        self.normalize_enabled = bool(state)
        self._apply_volume()

    def _on_default_view_toggled(self, state: int):
        self.default_10sec_view_enabled = bool(state)
        self.waveform.set_default_window_limit_ms(10 * 1000 if self.default_10sec_view_enabled else 0)
        if self.current_track_index >= 0:
            duration_ms = self.playlist[self.current_track_index].duration_s * 1000
            self.waveform.set_track_duration(duration_ms)
            self.zoom_slider.setValue(self.waveform.get_zoom_level())

    def _on_dynamic_waveform_toggled(self, state: int):
        self.dynamic_waveform_enabled = bool(state)
        if self.current_track_index >= 0:
            self._schedule_full_waveform_request(force=True, immediate=True)

    def _apply_volume(self):
        base = self.volume_slider.value() / 100.0
        if not self.normalize_enabled or self.current_track_index < 0:
            self.audio_output.setVolume(base)
            return
        fp = self.playlist[self.current_track_index].filepath
        rms = self.track_rms.get(fp)
        if not rms or rms <= 0:
            self.audio_output.setVolume(base)
            return
        gain = self.target_rms / rms
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
            "- Visible timeline scale\n"
            "- Mouse wheel and slider zoom\n"
            "- Right-drag panning\n"
            "- Normalize toggle\n"
            "- Playlist save/load (.m3u)\n",
        )

    def closeEvent(self, event):
        self._playback_refresh_timer.stop()
        self._stop_loader()
        self.player.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MusicPlayer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
