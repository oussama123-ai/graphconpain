"""
tests/test_preprocessing.py
-----------------------------
Unit tests for preprocessing pipelines.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np


class TestAudioMFCC:
    def test_output_dim(self, tmp_path):
        """Test that MFCC extraction produces (T, 65) arrays."""
        import soundfile as sf
        import numpy as np
        audio_path = tmp_path / "test.wav"
        sr = 16000
        duration = 2  # seconds
        y = np.random.randn(sr * duration).astype(np.float32) * 0.1
        try:
            sf.write(str(audio_path), y, sr)
            from data.preprocessing.audio_mfcc import extract_features
            feats = extract_features(audio_path, sr=sr, video_fps=30)
            assert feats.shape[1] == 65, f"Expected 65 features, got {feats.shape[1]}"
            assert feats.shape[0] > 0
        except ImportError:
            pytest.skip("soundfile not installed")


class TestPhysioPreprocessing:
    def test_sliding_windows(self):
        from data.preprocessing.physiological import sliding_windows
        signal  = np.random.randn(1000)
        windows = sliding_windows(signal, window=250, overlap=0.5)
        assert windows.shape[1] == 250
        assert windows.shape[0] > 0


class TestFacialSmoothing:
    def test_temporal_smooth(self):
        from data.preprocessing.facial_au import temporal_smooth
        aus = np.random.rand(90, 17).astype(np.float32)
        smoothed = temporal_smooth(aus)
        assert smoothed.shape == aus.shape
        # Smoothing should reduce variance slightly
        assert smoothed.var() <= aus.var() + 1e-5

    def test_interpolate_missing(self):
        from data.preprocessing.facial_au import interpolate_missing
        aus = np.random.rand(30, 17).astype(np.float32)
        aus[5:8] = 0.0   # simulate missing frames
        filled = interpolate_missing(aus)
        assert filled.shape == aus.shape
