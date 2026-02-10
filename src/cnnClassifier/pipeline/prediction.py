import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class PredictionPipeline:
    # Class-level cache
    _cancer_model = None
    _gate_model = None
    _optimal_threshold = None
    _gate_threshold = None

    def __init__(self, filename):
        self.filename = filename
        
        # Load models ONCE
        if PredictionPipeline._cancer_model is None:
            PredictionPipeline._cancer_model = self._load_cancer_model()
        
        if PredictionPipeline._gate_model is None:
            PredictionPipeline._gate_model = self._load_gate_model()
        
        # Load optimal thresholds from scores.json
        if PredictionPipeline._optimal_threshold is None:
            self._load_thresholds()

    def _load_thresholds(self):
        """Load optimal thresholds from evaluation results"""
        try:
            with open("scores.json", "r") as f:
                scores = json.load(f)
                PredictionPipeline._optimal_threshold = scores.get("test_optimal_threshold", 0.5)
                print(f"✅ Loaded optimal cancer threshold: {PredictionPipeline._optimal_threshold:.4f}")
        except FileNotFoundError:
            print("⚠️  scores.json not found. Using default threshold 0.5")
            PredictionPipeline._optimal_threshold = 0.5
        
        # Gate threshold (can be tuned separately if needed)
        PredictionPipeline._gate_threshold = 0.5

    def _load_cancer_model(self):
        model_path = os.path.join("model","model.h5")
        try:
            model = load_model(model_path)
            print("✅ Cancer model loaded successfully")
            return model
        except Exception as e:
            print(f"⚠️  Direct load failed: {e}")
            # Rebuild architecture
            backbone = tf.keras.applications.EfficientNetB0(
                input_shape=(224, 224, 3), weights=None, include_top=False
            )
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = backbone(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            model = tf.keras.Model(inputs, outputs)
            model.load_weights(model_path)
            print("✅ Cancer model weights loaded via rebuild")
            return model

    def _load_gate_model(self):
        gate_path = os.path.join("model","ct_gate_model.h5")
        model = load_model(gate_path, compile=False)
        print("✅ Gate model loaded successfully")
        return model

    def predict(self):
        # Load and preprocess image
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)  # 0-255 range
        img_array = np.expand_dims(img_array, axis=0)

        # ---------------------------------------------------------
        # PREPROCESSING (VERIFIED CORRECT)
        # ---------------------------------------------------------
        # Gate model: trained on 0-255 (from image_dataset_from_directory)
        img_for_gate = img_array  
        
        # Main model: trained on 0-1 (from ImageDataGenerator with rescale)
        img_for_main = img_array / 255.0

        # ---------------------------------------------------------
        # STEP 1: GATE KEEPER CHECK
        # ---------------------------------------------------------
        gate_prob_non_ct = float(self._gate_model.predict(img_for_gate, verbose=0)[0][0])
        gate_ct_confidence = 1.0 - gate_prob_non_ct

        print(f"[Gate] Non-CT Probability: {gate_prob_non_ct:.4f} | CT Confidence: {gate_ct_confidence:.4f}")

        if gate_prob_non_ct > self._gate_threshold:
            print(f"[Gate] REJECTED - Not a CT scan")
            return {
                "is_ct": False,
                "label": "Not a CT Scan",
                "probability": gate_prob_non_ct,
                "gate_p": gate_prob_non_ct,
                "cancer_p": None
            }

        # ---------------------------------------------------------
        # STEP 2: CANCER CLASSIFICATION (WITH OPTIMAL THRESHOLD)
        # ---------------------------------------------------------
        pred = self._cancer_model.predict(img_for_main, verbose=0)
        p_cancer = float(pred[0][0])  # Raw probability

        # Use OPTIMAL threshold from evaluation
        if p_cancer > self._optimal_threshold:
            final_label = "Adenocarcinoma"
            final_prob = p_cancer
        else:
            final_label = "Normal"
            final_prob = 1.0 - p_cancer

        print(f"[Cancer Model] Raw Probability: {p_cancer:.4f}")
        print(f"[Decision] Using threshold {self._optimal_threshold:.4f} → {final_label} (Confidence: {final_prob:.4f})")
        
        return {
            "is_ct": True,
            "label": final_label,
            "probability": final_prob,
            "gate_p": gate_prob_non_ct,
            "cancer_p00": p_cancer,
            "threshold_used": self._optimal_threshold
        }