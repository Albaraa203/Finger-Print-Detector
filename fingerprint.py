"""
================================================================================
  FINGERPRINT VERIFICATION SYSTEM
  Project : SOCOFing Fingerprint Verification (Siamese / Triplet-Loss Network)
  Author  : Graduation Project — Fingerprint Verification Team

  Supports  : .keras  AND  .h5  model files (auto-detected)
  Features  : Verification  |  Evaluation  |  Confusion Matrix
              ROC Curve     |  Full Metrics |  Professional PySide6 GUI

EXPECTED FILES (same folder as this script)
────────────────────────────────────────────
    fingernet_v2.keras   OR   fingernet_v2.h5   <- trained Keras encoder
    image_data.npy          <- preprocessed images   (N, 96, 96)
    features_data.npy       <- metadata               (N, 5)
    test.py                 <- this script

USAGE
─────
    python test.py             # CLI demo + GUI
    python test.py --no-gui    # CLI only
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import argparse
import traceback
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  THIRD-PARTY  (scientific)
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

# Matplotlib — Agg backend so saving always works headlessly.
# The GUI renders plots itself via in-memory PNG → QPixmap.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ─────────────────────────────────────────────────────────────────────────────
#  GUI  (PySide6 preferred, fallback to PyQt6, then graceful skip)
# ─────────────────────────────────────────────────────────────────────────────
_GUI_AVAILABLE = False
_QT_BACKEND = "none"

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
        QTabWidget, QFrame, QStatusBar, QMessageBox,
    )
    from PySide6.QtGui import QPixmap, QImage, QFont
    from PySide6.QtCore import Qt, QThread, Signal
    _GUI_AVAILABLE = True
    _QT_BACKEND = "PySide6"
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
            QTabWidget, QFrame, QStatusBar, QMessageBox,
        )
        from PyQt6.QtGui import QPixmap, QImage, QFont
        from PyQt6.QtCore import Qt, QThread
        from PyQt6.QtCore import pyqtSignal as Signal
        _GUI_AVAILABLE = True
        _QT_BACKEND = "PyQt6"
    except ImportError:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  GLOBAL SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_model_path() -> Path:
    """Prefer .keras; fall back to .h5 if only that exists."""
    for name in ("fingernet_v2.keras", "fingernet_v2.h5"):
        p = _SCRIPT_DIR / name
        if p.exists():
            return p
    return _SCRIPT_DIR / "fingernet_v2.keras"   # will fail gracefully later


MODEL_PATH    = _find_model_path()
IMAGES_PATH   = _SCRIPT_DIR / "image_data.npy"
FEATURES_PATH = _SCRIPT_DIR / "features_data.npy"

IMG_HEIGHT   = 96
IMG_WIDTH    = 96
IMG_CHANNELS = 1
INPUT_SHAPE  = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

THRESHOLD = 0.75    # Euclidean distance < THRESHOLD  →  SAME PERSON

COL_ID     = 0
COL_GENDER = 1
COL_HAND   = 2
COL_FINGER = 3
COL_TYPE   = 4

# Dark colour palette used by all plots and the GUI
_C = {
    "bg":    "#0F1117",
    "panel": "#1A1D27",
    "accent":"#00D4AA",
    "warn":  "#FF6B6B",
    "text":  "#E8EAF0",
    "muted": "#6B7280",
    "green": "#2ECC71",
    "red":   "#E74C3C",
}


# ═════════════════════════════════════════════════════════════════════════════
#  1.  MODEL LOADING  (.keras AND .h5)
# ═════════════════════════════════════════════════════════════════════════════

def load_model_file(model_path) -> tf.keras.Model:
    """
    Load the trained Keras encoder with full Lambda-layer resilience.

    ─────────────────────────────────────────────────────────────────────────
    ROOT CAUSE & STRATEGY
    ─────────────────────────────────────────────────────────────────────────
    Keras serialises Lambda layers by encoding their Python bytecode with
    marshal+base64 and reconstructing it via eval()/exec() at load time.
    That reconstruction runs in an EMPTY namespace — so any name the lambda
    body references (including "tf", "K", "backend") will be undefined, even
    if you passed custom_objects={"tf": tf}.

    custom_objects only resolves *layer class names* during JSON config
    parsing; it is never injected into the Lambda closure's eval scope.

    WHY enable_unsafe_deserialization() DOESN'T HELP:
    It only removes the security gate on pickle-based objects.  It does
    nothing about the missing namespace inside a Lambda closure eval.

    FIX STRATEGY (three-stage cascade):
    ─────────────────────────────────────────────────────────────────────────
    Stage 1 — Monkey-patch the Lambda eval namespace
        We subclass Lambda before loading and override _deserialize_function
        so the reconstructed closure has access to tf/K.  This fixes loads
        where TF version ≤ 2.15 (old Keras 2 serialisation path).

    Stage 2 — Post-load Lambda replacement
        After loading, walk every layer.  Any Lambda whose call fails is
        replaced by an equivalent stateless Keras operation (L2Normalisation
        or a custom layer).  Weights are unaffected because Lambda layers are
        stateless.

    Stage 3 — Architecture reconstruction (last resort)
        If the file cannot be parsed at all (version mismatch, missing ops),
        we reconstruct the encoder architecture from scratch and load ONLY
        the weights (by_name=True so layer renames don't matter).

    Supports: .keras (Keras v3 native) and .h5 (HDF5 legacy), auto-detected.
    """
    model_path = Path(model_path)

    # ── Locate the file (try complementary extension if primary missing) ──────
    if not model_path.is_file():
        alt = model_path.with_suffix(
            ".h5" if model_path.suffix == ".keras" else ".keras"
        )
        if alt.is_file():
            print(f"[INFO] '{model_path.name}' not found — using '{alt.name}'.")
            model_path = alt
        else:
            sys.exit(
                f"\n[ERROR] Model file not found:\n  {model_path}\n"
                f"  Also tried: {alt}\n"
                f"Place a .keras or .h5 model in the same folder as test.py.\n"
            )

    ext = model_path.suffix.lower()
    print(f"[INFO] Loading model ({ext}) from: {model_path}")

    encoder = None

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 1 — Lambda namespace patch + standard load
    # ══════════════════════════════════════════════════════════════════════════
    # We register a patched Lambda class that injects tf/K into the closure
    # namespace at deserialization time.

    import builtins, types, marshal, base64

    class _PatchedLambda(tf.keras.layers.Lambda):
        """
        Drop-in Lambda replacement that reconstructs closures with tf/K in scope.

        Keras stores the Lambda function as:
            {"class_name": "Lambda",
             "config": {"function": {"class_name": "function",
                                     "config": <base64-bytecode>}}}
        We intercept _deserialize_function (Keras 2) or from_config (Keras 3)
        and eval the bytecode with a pre-populated globals dict.
        """

        @classmethod
        def _safe_globals(cls) -> dict:
            """Build a globals dict that satisfies any likely Lambda body."""
            import keras.backend as _K
            g = {
                "__builtins__": builtins,
                "tf":           tf,
                "tensorflow":   tf,
                "K":            _K,
                "backend":      _K,
                "np":           np,
                "numpy":        np,
            }
            # Also expose tf.math / tf.nn directly
            g["math"]   = tf.math
            g["nn"]     = tf.nn
            return g

        @classmethod
        def from_config(cls, config):
            """
            Override from_config to catch and re-eval any broken Lambda closures.
            Works for both Keras 2 (function stored as bytecode) and Keras 3
            (function stored as source string via dill/cloudpickle).
            """
            fn_cfg = config.get("function")
            if fn_cfg is None:
                return super().from_config(config)

            # Keras 2 path: fn is a list [bytecode_b64, closure_info, defaults]
            if isinstance(fn_cfg, (list, tuple)) and len(fn_cfg) >= 1:
                try:
                    raw_code = fn_cfg[0]
                    code_bytes = base64.b64decode(raw_code)
                    code_obj   = marshal.loads(code_bytes)
                    fn = types.FunctionType(code_obj, cls._safe_globals())
                    config = dict(config)
                    config["function"] = fn
                except Exception:
                    pass   # fall through to super()

            # Keras 3 path: fn is a dict with "config" -> source string
            elif isinstance(fn_cfg, dict):
                source = (fn_cfg.get("config") or fn_cfg.get("value") or "")
                if isinstance(source, str) and ("tf." in source or "K." in source):
                    try:
                        g = cls._safe_globals()
                        exec(source, g)   # defines the function in g
                        # The last defined callable in g is our function
                        fn = next(
                            (v for k, v in reversed(list(g.items()))
                             if callable(v) and k not in cls._safe_globals()),
                            None,
                        )
                        if fn is not None:
                            config = dict(config)
                            config["function"] = fn
                    except Exception:
                        pass

            try:
                return super().from_config(config)
            except Exception:
                # Absolute fallback: create an identity Lambda
                print("[WARNING] Lambda config unparseable; substituting "
                      "L2-normalise (axis=1). Verify this matches training.")
                return cls(lambda x: tf.math.l2_normalize(x, axis=1))

    # Register the patched class so load_model uses it for every Lambda
    _load_custom = {
        "Lambda":       _PatchedLambda,
        "tf":           tf,
        "tensorflow":   tf,
    }

    try:
        # enable_unsafe_deserialization is needed for .h5 with pickle payloads
        if hasattr(tf.keras.config, "enable_unsafe_deserialization"):
            tf.keras.config.enable_unsafe_deserialization()

        encoder = tf.keras.models.load_model(
            str(model_path),
            custom_objects=_load_custom,
            compile=False,
        )

        # Warm-up pass — this is where Lambda errors surface
        _dummy = tf.zeros((1,) + INPUT_SHAPE, dtype=tf.float32)
        encoder(_dummy, training=False)
        print("[INFO] Stage 1 load: OK")

    except Exception as stage1_err:
        print(f"[INFO] Stage 1 load failed: {stage1_err}")
        print("[INFO] Attempting Stage 2: post-load Lambda replacement ...")
        encoder = None

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 2 — Post-load Lambda replacement
    #  Walk every layer in the loaded model; replace broken Lambdas with a
    #  native equivalent layer that never needs deserialization.
    # ══════════════════════════════════════════════════════════════════════════
    if encoder is None:
        try:
            # Load silently, ignoring Lambda call errors for now
            if hasattr(tf.keras.config, "enable_unsafe_deserialization"):
                tf.keras.config.enable_unsafe_deserialization()

            _raw = tf.keras.models.load_model(
                str(model_path),
                custom_objects=_load_custom,
                compile=False,
            )
            encoder = _patch_lambda_layers(_raw)

            _dummy = tf.zeros((1,) + INPUT_SHAPE, dtype=tf.float32)
            encoder(_dummy, training=False)
            print("[INFO] Stage 2 load: OK (Lambda layers patched)")

        except Exception as stage2_err:
            print(f"[INFO] Stage 2 load failed: {stage2_err}")
            print("[INFO] Attempting Stage 3: weights-only reconstruction ...")
            encoder = None

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 3 — Architecture reconstruction + weights-only load
    #  Build the standard FingerprintNet encoder from scratch; load weights
    #  by layer name so minor naming differences don't break it.
    # ══════════════════════════════════════════════════════════════════════════
    if encoder is None:
        try:
            print("[INFO] Rebuilding encoder architecture from scratch ...")
            encoder = _build_encoder_architecture()

            # Build the model so weight shapes are defined
            _dummy = tf.zeros((1,) + INPUT_SHAPE, dtype=tf.float32)
            encoder(_dummy, training=False)

            encoder.load_weights(str(model_path), by_name=True, skip_mismatch=True)
            encoder(_dummy, training=False)
            print("[INFO] Stage 3 load: OK (architecture rebuilt, weights loaded)")

        except Exception as stage3_err:
            print(f"\n[ERROR] All three loading strategies failed.")
            print(f"  Stage 3 error: {stage3_err}")
            print("\n  Diagnosis checklist:")
            print("  1. Is the .h5 file the encoder or the FULL siamese network?")
            print("     If full siamese, extract the encoder sub-model first:")
            print("       full = tf.keras.models.load_model(..., compile=False)")
            print("       encoder = full.get_layer('encoder')  # or by index")
            print("  2. Check TF version match: model was likely saved with TF 2.x")
            print(f"     Current TF: {tf.__version__}")
            print("  3. Try: python -c \"import tensorflow as tf; "
                  "m=tf.keras.models.load_model('fingernet_v2.h5', compile=False); "
                  "print([l.name for l in m.layers])\"")
            traceback.print_exc()
            sys.exit(1)

    # ── Validate output shape ─────────────────────────────────────────────────
    expected = (None,) + INPUT_SHAPE
    actual   = tuple(encoder.input_shape)
    if actual != expected:
        print(
            f"[WARNING] Model input shape {actual} != expected {expected}.\n"
            f"          Preprocessing will attempt to match."
        )

    print(f"[INFO] Encoder ready  |  input={encoder.input_shape}"
          f"  output={encoder.output_shape}")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS for load_model_file()
# ─────────────────────────────────────────────────────────────────────────────

def _patch_lambda_layers(model: tf.keras.Model) -> tf.keras.Model:
    """
    Walk a loaded model and replace every Lambda layer that raises on call
    with a stateless L2Normalisation layer.

    This works by rebuilding the model as a new Functional graph, swapping
    broken Lambdas for known-good replacements.  Weights are preserved
    because Lambda layers are stateless.

    Parameters
    ----------
    model : tf.keras.Model  (possibly contains broken Lambda layers)

    Returns
    -------
    tf.keras.Model  with working layers
    """

    class _L2NormalizeLayer(tf.keras.layers.Layer):
        """Native Keras layer that applies L2 normalisation on axis=1."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def call(self, x):
            return tf.math.l2_normalize(x, axis=1)
        def get_config(self):
            return super().get_config()

    # Test whether the model as-is already works
    try:
        _d = tf.zeros((1,) + INPUT_SHAPE, dtype=tf.float32)
        model(_d, training=False)
        return model   # already fine — no patching needed
    except Exception:
        pass

    # For Sequential models: replace the broken layer in-place on the config
    if isinstance(model, tf.keras.Sequential):
        return _patch_sequential(model, _L2NormalizeLayer)

    # For Functional models: clone_model with layer_fn swap
    if hasattr(model, "_is_graph_network") and model._is_graph_network:
        try:
            return _patch_functional(model, _L2NormalizeLayer)
        except Exception:
            pass

    # Fallback: try setting a working call on each Lambda
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Lambda):
            try:
                _d2 = tf.zeros((2, 128), dtype=tf.float32)
                layer(_d2)
            except Exception:
                # Patch the call in-place
                layer.function = lambda x: tf.math.l2_normalize(x, axis=1)
                print(f"[INFO] Patched Lambda layer '{layer.name}' in-place.")

    return model


def _patch_sequential(model, L2NormLayer):
    """Rebuild a Sequential model replacing Lambda → L2NormLayer."""
    new_model = tf.keras.Sequential(name=model.name)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Lambda):
            try:
                _d = tf.zeros((2, 128), dtype=tf.float32)
                layer(_d)
                new_model.add(layer)   # works fine
            except Exception:
                print(f"[INFO] Replacing Lambda '{layer.name}' with L2NormalizeLayer.")
                new_model.add(L2NormLayer(name=layer.name + "_patched"))
        else:
            new_model.add(layer)

    # Copy weights (Sequential add preserves references to original layers)
    return new_model


def _patch_functional(model, L2NormLayer):
    """
    Rebuild a Functional model substituting broken Lambdas.
    Uses tf.keras.models.clone_model with a custom clone_function.
    """
    def _clone_fn(layer):
        if not isinstance(layer, tf.keras.layers.Lambda):
            return layer.__class__.from_config(layer.get_config())
        # Test the Lambda
        try:
            test_in = tf.keras.Input(shape=layer.input_spec[0].shape[1:]
                                     if layer.input_spec else (128,))
            layer(test_in)
            return layer   # keep it
        except Exception:
            print(f"[INFO] Replacing Lambda '{layer.name}' with L2NormalizeLayer.")
            return L2NormLayer(name=layer.name + "_patched")

    try:
        new_model = tf.keras.models.clone_model(model, clone_function=_clone_fn)
        new_model.set_weights(model.get_weights())
        return new_model
    except Exception:
        return model   # give up, return original


def _build_encoder_architecture(embedding_size: int = 128) -> tf.keras.Model:
    inp = tf.keras.Input(shape=INPUT_SHAPE, name="input_1")

    # Match the 128-filter configuration from training
    x = tf.keras.layers.Conv2D(128, 5, activation='relu', kernel_initializer='he_uniform')(inp)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)

    # Match the 512-unit configuration from training
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(embedding_size)(x)

    out = tf.keras.layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name="normalize")(x)
    return tf.keras.Model(inputs=inp, outputs=out)


# ═════════════════════════════════════════════════════════════════════════════
#  2.  DATASET LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(images_path, features_path):
    """
    Load preprocessed .npy arrays from Part I.

    images   : (N, 96, 96)   float32 in [0, 1]
    features : (N, 5)        bytes-strings  [ID, Gender, Hand, Finger, Type]
    """
    for path, label in [
        (Path(images_path),   "image_data.npy"),
        (Path(features_path), "features_data.npy"),
    ]:
        if not path.is_file():
            sys.exit(
                f"\n[ERROR] Dataset file not found:\n  {path}\n"
                f"Place '{label}' in the same folder as test.py.\n"
            )

    images   = np.load(str(images_path),   allow_pickle=True)
    features = np.load(str(features_path), allow_pickle=True)

    if images.ndim not in (3, 4):
        sys.exit(f"\n[ERROR] image_data.npy: unexpected shape {images.shape}.")
    if features.ndim != 2 or features.shape[1] != 5:
        sys.exit(f"\n[ERROR] features_data.npy: unexpected shape {features.shape}.")
    if images.shape[0] != features.shape[0]:
        sys.exit("\n[ERROR] images/features sample count mismatch.")

    # Strip trailing channel dim if present
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]

    print(f"\n{'─'*55}")
    print(f"  DATASET SUMMARY")
    print(f"{'─'*55}")
    print(f"  Total samples  : {images.shape[0]}")
    print(f"  Image shape    : {images.shape}")
    print(f"  Pixel range    : [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Columns        : [ID, Gender, Hand, Finger, ImageType]")
    print(f"  Unique persons : {len(np.unique(features[:, COL_ID]))}")
    print(f"{'─'*55}\n")

    return images, features


# ═════════════════════════════════════════════════════════════════════════════
#  3.  PREPROCESSING  (mirrors Part I exactly)
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Reproduce the preprocessing pipeline from Part I.

    Steps
    ─────
    1. Grayscale  (handles 3-ch input from cv2.imread)
    2. float32 in [0, 1]  (matches tf.image.convert_image_dtype)
    3. Resize to (96 × 96)  (matches cv2.resize fallback in Part I)
    4. Channel dim  → (96, 96, 1)
    5. Batch dim    → (1, 96, 96, 1)

    Accepted input shapes: (H,W)  |  (H,W,1)  |  (H,W,3)
    Returns: (1, 96, 96, 1)  float32
    """
    # Step 1 – grayscale
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]

    # Step 2 – float32 normalisation
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

    # Step 3 – resize
    if img.shape != (IMG_HEIGHT, IMG_WIDTH):
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Steps 4 & 5 – add dims
    img = np.expand_dims(img, axis=-1)   # (96, 96, 1)
    img = np.expand_dims(img, axis=0)    # (1, 96, 96, 1)
    return img


# ═════════════════════════════════════════════════════════════════════════════
#  4.  METADATA HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def decode_feature(feat_row: np.ndarray) -> dict:
    """Convert a feature row (bytes-strings) to a human-readable dict."""
    def _d(val):
        s = (val.decode("utf-8", errors="replace")
             if isinstance(val, (bytes, np.bytes_)) else str(val))
        return s.split("\\")[-1].strip()

    return {
        "id":         _d(feat_row[COL_ID]),
        "gender":     _d(feat_row[COL_GENDER]),
        "hand":       _d(feat_row[COL_HAND]),
        "finger":     _d(feat_row[COL_FINGER]),
        "image_type": _d(feat_row[COL_TYPE]),
    }


def same_person(feat_a: np.ndarray, feat_b: np.ndarray) -> bool:
    """True if both feature rows share the same person ID (metadata label)."""
    def _id(f):
        v = f[COL_ID]
        s = (v.decode("utf-8", errors="replace")
             if isinstance(v, (bytes, np.bytes_)) else str(v))
        return s.split("\\")[-1].strip()
    return _id(feat_a) == _id(feat_b)


# ═════════════════════════════════════════════════════════════════════════════
#  5.  CORE VERIFICATION LOGIC
# ═════════════════════════════════════════════════════════════════════════════

def get_embedding(encoder: tf.keras.Model, raw_img: np.ndarray) -> np.ndarray:
    """Run one (96,96) image through the encoder → embedding vector."""
    tensor = preprocess_image(raw_img)           # (1, 96, 96, 1)
    emb    = encoder.predict(tensor, verbose=0)  # (1, emb_size)
    return emb[0]                                # (emb_size,)


def euclidean_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Euclidean distance on the L2-normalised unit sphere → [0, 2]."""
    return float(np.linalg.norm(emb_a - emb_b))


def distance_to_similarity(distance: float, max_dist: float = 2.0) -> float:
    """Map distance [0, 2] → similarity score [0, 1]."""
    return max(0.0, 1.0 - distance / max_dist)


def compare_fingerprints(encoder, img_a, img_b, feat_a, feat_b,
                          threshold=THRESHOLD) -> dict:
    """
    Full pipeline: preprocess → embed → compare → decide.

    Returns dict with: embedding_a/b, distance, similarity,
    prediction, ground_truth, correct, meta_a/b.
    """
    emb_a = get_embedding(encoder, img_a)
    emb_b = get_embedding(encoder, img_b)
    dist  = euclidean_distance(emb_a, emb_b)
    sim   = distance_to_similarity(dist)
    pred  = "SAME PERSON" if dist < threshold else "DIFFERENT PERSON"
    gt    = "SAME PERSON" if same_person(feat_a, feat_b) else "DIFFERENT PERSON"
    return {
        "embedding_a":  emb_a,
        "embedding_b":  emb_b,
        "distance":     dist,
        "similarity":   sim,
        "prediction":   pred,
        "ground_truth": gt,
        "correct":      pred == gt,
        "meta_a":       decode_feature(feat_a),
        "meta_b":       decode_feature(feat_b),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  6.  BATCH EMBEDDING HELPER  (vectorised — fast)
# ═════════════════════════════════════════════════════════════════════════════

def batch_embeddings(encoder: tf.keras.Model, images: np.ndarray,
                     batch_size: int = 64) -> np.ndarray:
    """
    Compute embeddings for an (N, 96, 96) array in mini-batches.

    Returns np.ndarray of shape (N, embedding_size).
    """
    arr = images.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]      # (N, 96, 96, 1)
    if arr.max() > 1.0:
        arr /= 255.0

    results = []
    for start in range(0, len(arr), batch_size):
        results.append(encoder.predict(arr[start:start + batch_size], verbose=0))
    return np.concatenate(results, axis=0)


# ═════════════════════════════════════════════════════════════════════════════
#  7.  FULL EVALUATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    encoder:   tf.keras.Model,
    images:    np.ndarray,
    features:  np.ndarray,
    n_pairs:   int   = 500,
    threshold: float = THRESHOLD,
    save_figs: bool  = True,
    verbose:   bool  = True,
) -> dict:
    """
    Evaluate the encoder on n_pairs balanced random pairs.

    Balanced sampling: n_pairs/2 same-person + n_pairs/2 different-person pairs.
    Embeddings are pre-computed once in batch for maximum speed.

    Parameters
    ----------
    encoder   : tf.keras.Model
    images    : np.ndarray   (N, 96, 96)
    features  : np.ndarray   (N, 5)
    n_pairs   : int          total test pairs  (default 500)
    threshold : float        decision boundary  (default 0.5)
    save_figs : bool         save confusion_matrix.png and roc_curve.png
    verbose   : bool         print full report

    Returns
    -------
    dict: y_true, y_pred, y_scores, distances,
          accuracy, precision, recall, f1, roc_auc,
          tn, fp, fn, tp, confusion_matrix,
          classification_report, fpr, tpr
    """
    print(f"\n[EVAL] Generating {n_pairs} test pairs  (threshold={threshold}) ...")

    # Pre-compute all embeddings at once (much faster than per-pair inference)
    print("[EVAL] Pre-computing embeddings for the full dataset ...")
    all_embs = batch_embeddings(encoder, images)

    ids        = features[:, COL_ID]
    unique_ids = np.unique(ids)
    multi_ids  = [uid for uid in unique_ids if np.sum(ids == uid) >= 2]

    half   = n_pairs // 2
    y_true = []
    dists  = []

    # ── Same-person pairs ──────────────────────────────────────────────────────
    if len(multi_ids) == 0:
        print("[EVAL WARNING] No person has >= 2 images; skipping same-person pairs.")
        half = 0
    for _ in range(half):
        pid  = multi_ids[np.random.randint(len(multi_ids))]
        pool = np.where(ids == pid)[0]
        i, j = np.random.choice(pool, size=2, replace=False)
        dists.append(float(np.linalg.norm(all_embs[i] - all_embs[j])))
        y_true.append(1)   # label 1 = same person

    # ── Different-person pairs ─────────────────────────────────────────────────
    for _ in range(n_pairs - half):
        i    = np.random.randint(len(images))
        diff = np.where(ids != ids[i])[0]
        j    = int(np.random.choice(diff)) if len(diff) else (i + 1) % len(images)
        dists.append(float(np.linalg.norm(all_embs[i] - all_embs[j])))
        y_true.append(0)   # label 0 = different person

    dists  = np.array(dists)
    y_true = np.array(y_true)

    # Binary prediction: distance < threshold → predicted same (1)
    y_pred   = (dists < threshold).astype(int)

    # Similarity score for ROC (higher = more similar)
    y_scores = np.clip(1.0 - dists / 2.0, 0.0, 1.0)

    # ── sklearn metrics ────────────────────────────────────────────────────────
    cm         = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, int(cm[0, 0])))

    acc        = accuracy_score(y_true, y_pred)
    prec       = precision_score(y_true, y_pred, zero_division=0)
    rec        = recall_score(y_true, y_pred, zero_division=0)
    f1         = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc    = auc(fpr, tpr)
    cls_rep    = classification_report(
        y_true, y_pred,
        target_names=["Different", "Same"],
        zero_division=0,
    )

    if verbose:
        sep = "=" * 60; thin = "-" * 60
        print(f"\n{sep}")
        print(f"  EVALUATION REPORT  ({n_pairs} pairs, threshold={threshold})")
        print(f"{sep}")
        print(f"  Accuracy   : {acc  * 100:.2f}%")
        print(f"  Precision  : {prec * 100:.2f}%")
        print(f"  Recall     : {rec  * 100:.2f}%")
        print(f"  F1-Score   : {f1   * 100:.2f}%")
        print(f"  ROC AUC    : {roc_auc:.4f}")
        print(f"{thin}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"{thin}")
        print(cls_rep)
        print(f"{sep}\n")

    if save_figs:
        _plot_confusion_matrix(cm, save=True)
        _plot_roc_curve(fpr, tpr, roc_auc, save=True)

    return {
        "y_true": y_true, "y_pred": y_pred, "y_scores": y_scores,
        "distances": dists,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": roc_auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": cm, "classification_report": cls_rep,
        "fpr": fpr, "tpr": tpr,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  8.  PLOT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(cm: np.ndarray, save: bool = True,
                            path: str = "confusion_matrix.jpg") -> Figure:
    """Publication-quality confusion matrix. Returns Figure."""
    labels   = ["Different\n(Neg)", "Same\n(Pos)"]
    fig, ax  = plt.subplots(figsize=(6, 5), facecolor=_C["bg"])
    ax.set_facecolor(_C["panel"])

    cm_norm = cm.astype(float) / max(cm.sum(), 1)
    cmap    = plt.cm.get_cmap("YlGn")
    im      = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(labels, color=_C["text"], fontsize=11)
    ax.set_yticklabels(labels, color=_C["text"], fontsize=11,
                       rotation=90, va="center")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tc = "black" if cm_norm[i, j] > 0.5 else _C["text"]
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=tc)

    ax.set_xlabel("Predicted", color=_C["text"], fontsize=12, labelpad=8)
    ax.set_ylabel("True",      color=_C["text"], fontsize=12, labelpad=8)
    ax.set_title("Confusion Matrix", color=_C["accent"],
                 fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors=_C["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(_C["muted"])

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Proportion", color=_C["text"], fontsize=9)
    cb.ax.yaxis.set_tick_params(color=_C["text"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=_C["text"])

    fig.tight_layout()
    if save:
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=_C["bg"])
        print(f"[INFO] Saved: {path}")
    return fig


def _plot_roc_curve(fpr, tpr, roc_auc: float, save: bool = True,
                    path: str = "roc_curve.jpg") -> Figure:
    """Publication-quality ROC curve. Returns Figure."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=_C["bg"])
    ax.set_facecolor(_C["panel"])

    ax.plot(fpr, tpr, color=_C["accent"], lw=2.5,
            label=f"ROC  (AUC = {roc_auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color=_C["accent"])
    ax.plot([0, 1], [0, 1], color=_C["muted"], lw=1.5, linestyle="--",
            label="Random")

    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", color=_C["text"], fontsize=12, labelpad=8)
    ax.set_ylabel("True Positive Rate",  color=_C["text"], fontsize=12, labelpad=8)
    ax.set_title("ROC Curve — Fingerprint Verification",
                 color=_C["accent"], fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10,
              facecolor=_C["panel"], edgecolor=_C["muted"],
              labelcolor=_C["text"])
    ax.tick_params(colors=_C["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(_C["muted"])

    ax.text(0.97, 0.07, f"AUC = {roc_auc:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, fontweight="bold", color=_C["accent"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=_C["panel"],
                      edgecolor=_C["accent"], linewidth=1.5))
    fig.tight_layout()
    if save:
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=_C["bg"])
        print(f"[INFO] Saved: {path}")
    return fig


def visualize_results(img_a: np.ndarray, img_b: np.ndarray,
                       result: dict, save: bool = True) -> Figure:
    """Side-by-side fingerprint result figure. Returns Figure."""
    is_match   = result["prediction"] == "SAME PERSON"
    is_correct = result["correct"]
    distance   = result["distance"]
    similarity = result["similarity"]
    meta_a     = result["meta_a"]
    meta_b     = result["meta_b"]

    pred_color = _C["green"] if is_match   else _C["red"]
    eval_color = "#27AE60"   if is_correct else "#C0392B"
    BG         = _C["bg"]

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.suptitle("FINGERPRINT VERIFICATION SYSTEM  |  SOCOFing Dataset",
                 fontsize=15, fontweight="bold", color=_C["accent"], y=0.98)

    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.4],
                          hspace=0.35, wspace=0.3,
                          left=0.06, right=0.94, top=0.91, bottom=0.04)

    def _fp_panel(ax, img, title):
        ax.imshow(img.reshape(IMG_HEIGHT, IMG_WIDTH), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=_C["accent"], pad=10)
        ax.axis("off"); ax.set_facecolor(_C["panel"])

    def _meta_panel(ax, meta):
        ax.set_facecolor(_C["panel"]); ax.axis("off")
        txt = "\n".join(f"{k.replace('_',' ').title():<12}: {v}"
                        for k, v in meta.items())
        ax.text(0.5, 0.5, txt, ha="center", va="center",
                transform=ax.transAxes, fontsize=9.5, family="monospace",
                color=_C["text"],
                bbox=dict(boxstyle="round,pad=0.6", facecolor=_C["panel"],
                          edgecolor=_C["muted"], linewidth=1.2))

    _fp_panel(fig.add_subplot(gs[0, 0]), img_a, "FINGERPRINT  A")
    _meta_panel(fig.add_subplot(gs[1, 0]), meta_a)
    _fp_panel(fig.add_subplot(gs[0, 2]), img_b, "FINGERPRINT  B")
    _meta_panel(fig.add_subplot(gs[1, 2]), meta_b)

    # Central result panel
    ax_res = fig.add_subplot(gs[0, 1])
    ax_res.set_facecolor(_C["panel"]); ax_res.axis("off")

    verdict = "MATCH" if is_match else "NO MATCH"
    ax_res.text(0.5, 0.82, verdict, ha="center", va="center",
                transform=ax_res.transAxes, fontsize=18, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.55",
                          facecolor=pred_color, edgecolor="none"))

    bar_bg = ax_res.inset_axes([0.08, 0.55, 0.84, 0.10])
    bar_bg.set_xlim(0, 1); bar_bg.set_ylim(0, 1); bar_bg.axis("off")
    bar_bg.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02",
        facecolor=_C["muted"], edgecolor="none"))
    if similarity > 0.01:
        bar_fill = ax_res.inset_axes([0.08, 0.55, 0.84 * similarity, 0.10])
        bar_fill.set_xlim(0, 1); bar_fill.set_ylim(0, 1); bar_fill.axis("off")
        bar_fill.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02",
            facecolor=pred_color, edgecolor="none"))

    for y, txt, fs, col in [
        (0.47, f"Similarity : {similarity * 100:.1f}%", 11, _C["text"]),
        (0.37, f"Distance   : {distance:.4f}  (T={THRESHOLD})", 10, _C["muted"]),
        (0.22, f"Ground Truth : {result['ground_truth']}", 10, _C["text"]),
        (0.10, ("Correct" if is_correct else "Incorrect"), 10, eval_color),
        (0.01, f"Embedding dim : {len(result['embedding_a'])}", 8, _C["muted"]),
    ]:
        ax_res.text(0.5, y, txt, ha="center", va="center",
                    transform=ax_res.transAxes, fontsize=fs, color=col)

    # Distance bar
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_bar.set_facecolor(_C["panel"])
    ax_bar.barh(0, 2, color=_C["muted"], height=0.4, left=0, align="center")
    ax_bar.barh(0, min(distance, 2), color=pred_color, height=0.4,
                left=0, align="center")
    ax_bar.axvline(x=THRESHOLD, color="#E67E22", lw=2, linestyle="--")
    ax_bar.set_xlim(0, 2); ax_bar.set_ylim(-1, 1)
    ax_bar.set_title("Euclidean Distance  [0 -> 2]",
                     fontsize=9, color=_C["accent"], pad=4)
    ax_bar.text(distance, 0.45, f" {distance:.3f}",
                va="bottom", fontsize=9, color=pred_color, fontweight="bold")
    ax_bar.text(THRESHOLD, -0.65, f"T={THRESHOLD}",
                ha="center", fontsize=8, color="#E67E22")
    ax_bar.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax_bar.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax_bar.tick_params(axis="x", labelsize=8, colors=_C["text"])

    if save:
        fig.savefig("verification_result.jpg", dpi=130,
                    bbox_inches="tight", facecolor=BG)
        print("[INFO] Saved: verification_result.jpg")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  9.  PAIR SELECTION & CONSOLE PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def pick_random_pair(images, features, force_same=None):
    """Pick two dataset indices; optionally force same/different person."""
    n = len(images)
    if force_same is None:
        force_same = bool(np.random.randint(0, 2))

    if force_same:
        uids = np.unique(features[:, COL_ID])
        np.random.shuffle(uids)
        for pid in uids:
            idx = np.where(features[:, COL_ID] == pid)[0]
            if len(idx) >= 2:
                c = np.random.choice(idx, 2, replace=False)
                return int(c[0]), int(c[1])
        return np.random.randint(0, n), np.random.randint(0, n)

    i    = np.random.randint(0, n)
    diff = np.where(features[:, COL_ID] != features[i, COL_ID])[0]
    j    = int(np.random.choice(diff)) if len(diff) else (i + 1) % n
    return i, j


def print_results(result: dict, idx_a: int, idx_b: int) -> None:
    """Pretty-print a verification result to the console."""
    sep = "=" * 60; thin = "-" * 60
    print(f"\n{sep}")
    print(f"  FINGERPRINT VERIFICATION  —  RESULT")
    print(f"{sep}")
    for label, idx, mk in [("A", idx_a, "meta_a"), ("B", idx_b, "meta_b")]:
        m = result[mk]
        print(f"  Sample {label}  (index {idx})")
        for k, v in m.items():
            print(f"    {k.replace('_',' ').title():<12}: {v}")
        print(thin)
    d, s = result["distance"], result["similarity"]
    print(f"  Distance   : {d:.6f}  (threshold={THRESHOLD})")
    print(f"  Similarity : {s * 100:.2f}%")
    icon = "CORRECT" if result["correct"] else "INCORRECT"
    print(f"  Prediction : {result['prediction']}")
    print(f"  Ground Truth: {result['ground_truth']}")
    print(f"  Evaluation : {icon}")
    print(f"{sep}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  10.  PROFESSIONAL PySide6 / PyQt6 GUI
# ═════════════════════════════════════════════════════════════════════════════

if _GUI_AVAILABLE:

    def _fig_to_pixmap(fig: Figure) -> "QPixmap":
        """Render a matplotlib Figure to QPixmap via in-memory PNG."""
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="jpg", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        qimg = QImage.fromData(buf.read())
        return QPixmap.fromImage(qimg)


    class _VerifyWorker(QThread):
        """Background thread for single-pair verification."""
        result_ready   = Signal(dict, object, object)
        error_occurred = Signal(str)
        progress       = Signal(str)

        def __init__(self, encoder, img_a, img_b, feat_a, feat_b, threshold):
            super().__init__()
            self.encoder   = encoder
            self.img_a, self.img_b = img_a, img_b
            self.feat_a, self.feat_b = feat_a, feat_b
            self.threshold = threshold

        def run(self):
            try:
                self.progress.emit("Computing embeddings ...")
                res = compare_fingerprints(
                    self.encoder, self.img_a, self.img_b,
                    self.feat_a, self.feat_b, self.threshold)
                self.result_ready.emit(res, self.img_a, self.img_b)
            except Exception as e:
                self.error_occurred.emit(str(e))


    class _EvalWorker(QThread):
        """Background thread for batch evaluation."""
        finished = Signal(dict)
        progress = Signal(str)
        error    = Signal(str)

        def __init__(self, encoder, images, features, n_pairs, threshold):
            super().__init__()
            self.encoder, self.images, self.features = encoder, images, features
            self.n_pairs, self.threshold = n_pairs, threshold

        def run(self):
            try:
                self.progress.emit(f"Evaluating {self.n_pairs} pairs ...")
                res = evaluate_model(
                    self.encoder, self.images, self.features,
                    n_pairs=self.n_pairs, threshold=self.threshold,
                    save_figs=True, verbose=True)
                self.finished.emit(res)
            except Exception as e:
                self.error.emit(str(e))


    class _FPCard(QFrame):
        """Fingerprint image card with metadata label."""
        def __init__(self, title: str, parent=None):
            super().__init__(parent)
            self.setStyleSheet(f"""
                QFrame {{
                    background: {_C['panel']};
                    border: 1px solid {_C['muted']};
                    border-radius: 10px;
                }}
            """)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(10, 10, 10, 10)
            lay.setSpacing(6)

            ttl = QLabel(title)
            ttl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ttl.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
            ttl.setStyleSheet(f"color:{_C['accent']}; border:none;")

            self._img = QLabel()
            self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._img.setMinimumSize(180, 180)
            self._img.setStyleSheet("border:none;")

            self._meta = QLabel("No image loaded")
            self._meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._meta.setFont(QFont("Courier New", 9))
            self._meta.setStyleSheet(f"color:{_C['muted']}; border:none;")
            self._meta.setWordWrap(True)

            lay.addWidget(ttl)
            lay.addWidget(self._img)
            lay.addWidget(self._meta)

        def set_array(self, arr: np.ndarray):
            gray = (arr.reshape(IMG_HEIGHT, IMG_WIDTH) * 255).astype(np.uint8)
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
            pix  = QPixmap.fromImage(qimg).scaled(
                200, 200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self._img.setPixmap(pix)

        def set_meta(self, meta: dict):
            self._meta.setText(
                "\n".join(f"{k.replace('_',' ').title()}: {v}"
                           for k, v in meta.items()))
            self._meta.setStyleSheet(f"color:{_C['text']}; border:none;")

        def set_meta_text(self, text: str):
            self._meta.setText(text)


    # ── Main window ────────────────────────────────────────────────────────────
    _APP_STYLE = f"""
        QMainWindow, QWidget {{
            background-color: {_C['bg']};
            color: {_C['text']};
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }}
        QTabWidget::pane {{
            border: 1px solid {_C['muted']};
            border-radius: 8px;
            background: {_C['panel']};
        }}
        QTabBar::tab {{
            background: {_C['panel']};
            color: {_C['muted']};
            padding: 10px 22px;
            border-radius: 6px 6px 0 0;
            font-weight: 600;
        }}
        QTabBar::tab:selected {{
            background: {_C['bg']};
            color: {_C['accent']};
            border-bottom: 2px solid {_C['accent']};
        }}
        QPushButton {{
            background-color: {_C['accent']};
            color: #0F1117;
            border: none;
            border-radius: 8px;
            padding: 10px 18px;
            font-weight: 700;
            font-size: 13px;
            min-height: 36px;
        }}
        QPushButton:hover  {{ background-color: #00FFCC; }}
        QPushButton:pressed {{ background-color: #00AA88; }}
        QPushButton:disabled {{ background-color: {_C['muted']}; color:{_C['bg']}; }}
        QPushButton#sec {{
            background: transparent;
            color: {_C['accent']};
            border: 1px solid {_C['accent']};
        }}
        QPushButton#sec:hover {{ background: rgba(0,212,170,0.12); }}
        QProgressBar {{
            border: 1px solid {_C['muted']};
            border-radius: 6px;
            background: {_C['panel']};
            text-align: center;
            height: 14px;
        }}
        QProgressBar::chunk {{
            background: {_C['accent']};
            border-radius: 5px;
        }}
        QTextEdit {{
            background: {_C['panel']};
            color: {_C['text']};
            border: 1px solid {_C['muted']};
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            padding: 8px;
        }}
        QStatusBar {{
            background: {_C['panel']};
            color: {_C['muted']};
            border-top: 1px solid {_C['muted']};
        }}
    """

    class MainWindow(QMainWindow):
        def __init__(self, encoder, images, features):
            super().__init__()
            self.encoder  = encoder
            self.images   = images
            self.features = features

            self._img_a = self._img_b = None
            self._feat_a = self._feat_b = None
            self._vw = self._ew = None   # worker handles

            self.setWindowTitle(
                "Fingerprint Verification System  |  SOCOFing  |  Siamese Network")
            self.setMinimumSize(1100, 700)
            self.setStyleSheet(_APP_STYLE)
            self._build_ui()
            self._sb.showMessage("  Ready — load two fingerprints to begin.")

        # ── build ──────────────────────────────────────────────────────────────
        def _build_ui(self):
            cw  = QWidget(); self.setCentralWidget(cw)
            lay = QVBoxLayout(cw)
            lay.setContentsMargins(16, 12, 16, 8)
            lay.setSpacing(10)

            # Header
            hdr = QLabel("FINGERPRINT VERIFICATION SYSTEM")
            hdr.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hdr.setStyleSheet(f"color:{_C['accent']}; letter-spacing:2px;")
            lay.addWidget(hdr)

            sub = QLabel(
                f"Siamese / Triplet-Loss  |  SOCOFing  |  "
                f"Model: {MODEL_PATH.name}  |  Backend: {_QT_BACKEND}")
            sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sub.setStyleSheet(f"color:{_C['muted']}; font-size:11px;")
            lay.addWidget(sub)

            # Tabs
            tabs = QTabWidget()
            lay.addWidget(tabs, stretch=1)
            tabs.addTab(self._verify_tab(), "  Verify  ")
            tabs.addTab(self._eval_tab(),   "  Evaluate  ")
            tabs.addTab(self._about_tab(),  "  About  ")

            self._sb = QStatusBar()
            self.setStatusBar(self._sb)

        # ── Verify tab ─────────────────────────────────────────────────────────
        def _verify_tab(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(14, 14, 14, 14)
            lay.setSpacing(12)

            # Images row
            top = QHBoxLayout(); top.setSpacing(14)
            self._card_a = _FPCard("FINGERPRINT  A")
            self._card_b = _FPCard("FINGERPRINT  B")
            top.addWidget(self._card_a, 2)
            top.addWidget(self._result_panel(), 3)
            top.addWidget(self._card_b, 2)
            lay.addLayout(top, stretch=3)

            # Buttons
            br = QHBoxLayout(); br.setSpacing(10)
            self._b_la  = QPushButton("Load A")
            self._b_lb  = QPushButton("Load B")
            self._b_rs  = QPushButton("Random Same"); self._b_rs.setObjectName("sec")
            self._b_rd  = QPushButton("Random Diff"); self._b_rd.setObjectName("sec")
            self._b_run = QPushButton("RUN VERIFICATION")
            self._b_run.setEnabled(False)
            self._b_la.clicked.connect(lambda: self._load_file("A"))
            self._b_lb.clicked.connect(lambda: self._load_file("B"))
            self._b_rs.clicked.connect(lambda: self._load_rand(True))
            self._b_rd.clicked.connect(lambda: self._load_rand(False))
            self._b_run.clicked.connect(self._do_verify)
            for b in (self._b_la, self._b_lb, self._b_rs, self._b_rd, self._b_run):
                br.addWidget(b)
            lay.addLayout(br)

            self._vbar = QProgressBar()
            self._vbar.setRange(0, 0)
            self._vbar.setVisible(False)
            self._vbar.setFixedHeight(10)
            lay.addWidget(self._vbar)
            return w

        def _result_panel(self) -> QFrame:
            f   = QFrame()
            f.setStyleSheet(f"""QFrame{{
                background:{_C['panel']};
                border:1px solid {_C['muted']};
                border-radius:10px;}}""")
            lay = QVBoxLayout(f)
            lay.setContentsMargins(16, 14, 16, 14); lay.setSpacing(8)

            ttl = QLabel("VERIFICATION RESULT")
            ttl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ttl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            ttl.setStyleSheet(f"color:{_C['accent']}; border:none;")
            lay.addWidget(ttl)

            self._lv  = QLabel("—")   # verdict
            self._lv.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._lv.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
            self._lv.setStyleSheet(f"color:{_C['muted']};padding:8px;border:none;")
            lay.addWidget(self._lv)

            self._l_sim  = self._ml("Similarity",   "—")
            self._l_dist = self._ml("Distance",      "—")
            self._l_gt   = self._ml("Ground Truth",  "—")
            self._l_eval = self._ml("Evaluation",    "—")
            for lbl in (self._l_sim, self._l_dist, self._l_gt, self._l_eval):
                lay.addWidget(lbl)

            lay.addStretch()

            self._thumb = QLabel()
            self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._thumb.setMinimumHeight(70)
            self._thumb.setStyleSheet("border:none;")
            lay.addWidget(self._thumb)
            return f

        @staticmethod
        def _ml(key, val) -> QLabel:
            lbl = QLabel(f"<b style='color:{_C['muted']}'>{key}:</b>  {val}")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("border:none;")
            return lbl

        # ── Evaluate tab ────────────────────────────────────────────────────────
        def _eval_tab(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(14, 14, 14, 14); lay.setSpacing(12)

            br = QHBoxLayout(); br.setSpacing(10)
            self._b_e500  = QPushButton("Evaluate 500 pairs")
            self._b_e1000 = QPushButton("Evaluate 1000 pairs")
            self._b_e500.clicked.connect(lambda:  self._do_eval(500))
            self._b_e1000.clicked.connect(lambda: self._do_eval(1000))
            br.addWidget(self._b_e500); br.addWidget(self._b_e1000)
            lay.addLayout(br)

            self._ebar = QProgressBar()
            self._ebar.setRange(0, 0); self._ebar.setVisible(False)
            self._ebar.setFixedHeight(10)
            lay.addWidget(self._ebar)

            # Figures
            fr = QHBoxLayout(); fr.setSpacing(14)
            self._cm_lbl  = self._fig_ph("Confusion Matrix")
            self._roc_lbl = self._fig_ph("ROC Curve")
            fr.addWidget(self._cm_lbl, 1); fr.addWidget(self._roc_lbl, 1)
            lay.addLayout(fr, stretch=3)

            self._mtxt = QTextEdit()
            self._mtxt.setReadOnly(True)
            self._mtxt.setMaximumHeight(160)
            self._mtxt.setPlaceholderText("Run evaluation to see metrics ...")
            lay.addWidget(self._mtxt)
            return w

        @staticmethod
        def _fig_ph(text) -> QLabel:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"""
                background:{_C['panel']};
                border:1px dashed {_C['muted']};
                border-radius:8px;
                color:{_C['muted']}; font-size:12px; padding:40px;""")
            lbl.setMinimumHeight(280)
            return lbl

        # ── About tab ───────────────────────────────────────────────────────────
        def _about_tab(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(40, 24, 40, 24)
            te  = QTextEdit(); te.setReadOnly(True)
            te.setStyleSheet(f"background:{_C['panel']}; border:none;")
            te.setHtml(f"""
            <h2 style='color:{_C["accent"]}'>Fingerprint Verification System</h2>
            <p style='color:{_C["muted"]}'>
              Siamese / Triplet-Loss Network on SOCOFing Dataset
            </p><hr style='border-color:{_C["muted"]}'>
            <p>
              <b>Architecture:</b> Base encoder, L2-normalised embeddings<br>
              <b>Loss:</b> Triplet loss with margin<br>
              <b>Dataset:</b> SOCOFing — 600 subjects, 6 000 fingerprint images<br>
              <b>Decision:</b> Euclidean distance &lt; {THRESHOLD} → same person
            </p><hr style='border-color:{_C["muted"]}'>
            <p>
              <b>Model:</b> {MODEL_PATH}<br>
              <b>Images:</b> {IMAGES_PATH}<br>
              <b>Features:</b> {FEATURES_PATH}
            </p><hr style='border-color:{_C["muted"]}'>
            <p style='color:{_C["muted"]}; font-size:11px'>
              GUI: {_QT_BACKEND} &nbsp;·&nbsp; TensorFlow {tf.__version__}
            </p>
            """)
            lay.addWidget(te)
            return w

        # ── Actions ─────────────────────────────────────────────────────────────
        def _load_file(self, side: str):
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select Fingerprint {side}", str(Path.home()),
                "Images (*.bmp *.BMP *.png *.jpg *.jpeg *.tif)")
            if not path:
                return

            # =======================================================
            # EXACT PREPROCESSING FROM Untitled2.ipynb
            # =======================================================
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                QMessageBox.warning(self, "Load Error", f"Cannot read:\n{path}")
                return

            # 1. Resize to target size (96, 96)
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # 2. Convert to float32 and normalize
            arr = img_resized.astype(np.float32) / 255.0
            # =======================================================

            dummy = np.array([b"external", b"?", b"?", b"?", b"external"])

            if side == "A":
                self._img_a, self._feat_a = arr, dummy
                self._card_a.set_array(arr)
                self._card_a.set_meta_text(f"File: {Path(path).name}")
            else:
                self._img_b, self._feat_b = arr, dummy
                self._card_b.set_array(arr)
                self._card_b.set_meta_text(f"File: {Path(path).name}")

            self._check_ready()
            self._sb.showMessage(f"  Loaded fingerprint {side}: {Path(path).name}")

        def _load_rand(self, same: bool):
            ia, ib = pick_random_pair(self.images, self.features, force_same=same)
            self._img_a, self._feat_a = self.images[ia], self.features[ia]
            self._img_b, self._feat_b = self.images[ib], self.features[ib]
            self._card_a.set_array(self._img_a)
            self._card_a.set_meta(decode_feature(self._feat_a))
            self._card_b.set_array(self._img_b)
            self._card_b.set_meta(decode_feature(self._feat_b))
            self._check_ready()
            label = "same-person" if same else "different-person"
            self._sb.showMessage(f"  Loaded random {label} pair  (idx {ia}/{ib})")

        def _check_ready(self):
            self._b_run.setEnabled(
                self._img_a is not None and self._img_b is not None)

        def _do_verify(self):
            if self._vw and self._vw.isRunning():
                return
            self._b_run.setEnabled(False)
            self._vbar.setVisible(True)
            self._vw = _VerifyWorker(
                self.encoder,
                self._img_a, self._img_b,
                self._feat_a, self._feat_b,
                THRESHOLD)
            self._vw.result_ready.connect(self._on_verify)
            self._vw.error_occurred.connect(self._on_err)
            self._vw.progress.connect(lambda m: self._sb.showMessage(f"  {m}"))
            self._vw.start()

        def _on_verify(self, res: dict, img_a, img_b):
            self._vbar.setVisible(False)
            self._b_run.setEnabled(True)

            match = res["prediction"] == "SAME PERSON"
            col   = _C["green"] if match else _C["red"]
            self._lv.setText("MATCH" if match else "NO MATCH")
            self._lv.setStyleSheet(
                f"color:white; background:{col}; padding:8px; "
                f"border-radius:6px; font-weight:700; border:none;")

            def r(key, val, vc=None):
                c = vc or _C["text"]
                return (f"<b style='color:{_C['muted']}'>{key}:</b>"
                        f"  <span style='color:{c}'>{val}</span>")

            self._l_sim.setText(r("Similarity", f"{res['similarity']*100:.1f}%"))
            self._l_dist.setText(r("Distance", f"{res['distance']:.4f}  (T={THRESHOLD})"))
            self._l_gt.setText(r("Ground Truth", res["ground_truth"]))
            ec = _C["green"] if res["correct"] else _C["red"]
            self._l_eval.setText(r("Evaluation",
                                    "Correct" if res["correct"] else "Incorrect", ec))

            fig = visualize_results(img_a, img_b, res, save=True)
            pix = _fig_to_pixmap(fig).scaled(
                300, 100,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self._thumb.setPixmap(pix)
            plt.close(fig)
            self._sb.showMessage(
                f"  Done — {res['prediction']}  (dist={res['distance']:.4f})")

        def _do_eval(self, n: int):
            if self._ew and self._ew.isRunning():
                return
            for b in (self._b_e500, self._b_e1000):
                b.setEnabled(False)
            self._ebar.setVisible(True)
            self._mtxt.clear()
            self._ew = _EvalWorker(
                self.encoder, self.images, self.features, n, THRESHOLD)
            self._ew.finished.connect(self._on_eval)
            self._ew.error.connect(self._on_err)
            self._ew.progress.connect(lambda m: self._sb.showMessage(f"  {m}"))
            self._ew.start()

        def _on_eval(self, res: dict):
            self._ebar.setVisible(False)
            for b in (self._b_e500, self._b_e1000):
                b.setEnabled(True)

            # Confusion matrix
            fig_cm = _plot_confusion_matrix(res["confusion_matrix"], save=False)
            pix_cm = _fig_to_pixmap(fig_cm).scaled(
                self._cm_lbl.width() or 380, self._cm_lbl.height() or 300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self._cm_lbl.setPixmap(pix_cm)
            plt.close(fig_cm)

            # ROC curve
            fig_roc = _plot_roc_curve(res["fpr"], res["tpr"],
                                       res["roc_auc"], save=False)
            pix_roc = _fig_to_pixmap(fig_roc).scaled(
                self._roc_lbl.width() or 380, self._roc_lbl.height() or 300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self._roc_lbl.setPixmap(pix_roc)
            plt.close(fig_roc)

            # Metrics text
            n = len(res["y_true"])
            self._mtxt.setPlainText(
                f"Pairs    : {n}\n"
                f"Threshold: {THRESHOLD}\n"
                f"{'─'*38}\n"
                f"Accuracy : {res['accuracy']*100:.2f}%\n"
                f"Precision: {res['precision']*100:.2f}%\n"
                f"Recall   : {res['recall']*100:.2f}%\n"
                f"F1-Score : {res['f1']*100:.2f}%\n"
                f"ROC AUC  : {res['roc_auc']:.4f}\n"
                f"{'─'*38}\n"
                f"TP={res['tp']}  FP={res['fp']}"
                f"  FN={res['fn']}  TN={res['tn']}\n"
                f"{'─'*38}\n"
                f"{res['classification_report']}"
            )
            self._sb.showMessage(
                f"  Evaluation done  Acc={res['accuracy']*100:.1f}%"
                f"  F1={res['f1']*100:.1f}%  AUC={res['roc_auc']:.3f}")

        def _on_err(self, msg: str):
            self._vbar.setVisible(False)
            self._ebar.setVisible(False)
            self._b_run.setEnabled(True)
            for b in (self._b_e500, self._b_e1000):
                b.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Operation failed:\n{msg}")
            self._sb.showMessage(f"  Error: {msg}")


# ═════════════════════════════════════════════════════════════════════════════
#  11.  CLI DEMO + MAIN
# ═════════════════════════════════════════════════════════════════════════════

def _cli_demo(encoder, images, features):
    """Run two-pair demo and a 300-pair evaluation in the console."""
    for label, forced in [("SAME-PERSON", True), ("DIFFERENT-PERSON", False)]:
        print(f"\n{'='*60}")
        print(f"  TEST — {label} PAIR")
        print(f"{'='*60}")
        ia, ib = pick_random_pair(images, features, force_same=forced)
        res = compare_fingerprints(encoder,
                                    images[ia], images[ib],
                                    features[ia], features[ib])
        print_results(res, ia, ib)
        fig = visualize_results(images[ia], images[ib], res, save=True)
        plt.close(fig)

    print("\n[INFO] Running evaluation on 300 pairs ...")
    evaluate_model(encoder, images, features, n_pairs=300,
                   threshold=THRESHOLD, save_figs=True, verbose=True)


def main():
    parser = argparse.ArgumentParser(description="Fingerprint Verification System")
    parser.add_argument("--no-gui", action="store_true", help="Skip GUI, CLI only")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  FINGERPRINT VERIFICATION SYSTEM  ")
    print("  Siamese / Triplet-Loss  |  SOCOFing Dataset")
    print("=" * 60 + "\n")

    encoder          = load_model_file(MODEL_PATH)
    images, features = load_dataset(IMAGES_PATH, FEATURES_PATH)

    if args.no_gui or not _GUI_AVAILABLE:
        if not _GUI_AVAILABLE:
            print("[INFO] PySide6/PyQt6 not found — CLI mode.")
            print("       To enable GUI: pip install PySide6")
        _cli_demo(encoder, images, features)
        return

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("FingerprintVerification")
    app.setStyle("Fusion")
    win = MainWindow(encoder, images, features)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()