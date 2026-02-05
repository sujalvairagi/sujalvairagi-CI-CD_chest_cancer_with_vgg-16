import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

        # ✅ Load both models ONCE
        self.cancer_model = self._load_cancer_model()
        self.gate_model = self._load_gate_model()

    # -----------------------------
    # ✅ CANCER MODEL (EfficientNet)
    # -----------------------------
    def _load_cancer_model(self):
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3),
            weights=None,
            include_top=False
        )

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = backbone(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs, outputs, name="CancerClassifier")

        model_path = os.path.join("artifacts", "training", "model.h5")

        # ✅ Works for both saved weights/full model formats
        try:
            model.load_weights(model_path)
        except Exception:
            model = tf.keras.models.load_model(model_path)

        return model

    # -----------------------------
    # ✅ CT GATE MODEL
    # -----------------------------
    def _load_gate_model(self):
        gate_model_path = os.path.join("artifacts", "ct_gate", "ct_gate_model.h5")

        # Gate model must be a full saved model (not weights only)
        gate_model = tf.keras.models.load_model(gate_model_path, compile=False)
        return gate_model

    # -----------------------------
    # ✅ PREPROCESS IMAGE
    # -----------------------------
    def _preprocess(self):
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    # -----------------------------
    # ✅ MAIN PREDICT
    # -----------------------------
    def predict(self):
        img_array = self._preprocess()

        # ✅ 1) CT Gate first
        # gate_prob_non_ct = probability image is NON_CT (label=1)
        gate_prob_non_ct = float(self.gate_model.predict(img_array)[0][0])
        gate_prob_ct = 1 - gate_prob_non_ct

        # Reject non-CT
        if gate_prob_non_ct > 0.5:
            return {
                "label": "not_ct_scan",
                "confidence": gate_prob_non_ct,
                "ct_gate_label": "NON_CT",
                "ct_gate_confidence": gate_prob_non_ct,
                "gate_prob_non_ct": gate_prob_non_ct
            }

        # ✅ 2) Cancer model prediction (only if CT passed)
        pred = self.cancer_model.predict(img_array)
        p = float(pred[0][0])  # sigmoid probability of cancer

        # ✅ Confidence-threshold decision
        if p >= 0.70:
            return {
                "label": "cancer",
                "confidence": p,
                "ct_gate_label": "CT",
                "ct_gate_confidence": gate_prob_ct,
                "gate_prob_non_ct": gate_prob_non_ct
            }

        elif p <= 0.30:
            return {
                "label": "normal",
                "confidence": 1 - p,
                "ct_gate_label": "CT",
                "ct_gate_confidence": gate_prob_ct,
                "gate_prob_non_ct": gate_prob_non_ct
            }

        else:
            # uncertain zone
            # confidence measure: closer to 0.5 = less confident
            uncertainty_conf = 1 - abs(p - 0.5) * 2

            return {
                "label": "uncertain",
                "confidence": uncertainty_conf,
                "ct_gate_label": "CT",
                "ct_gate_confidence": gate_prob_ct,
                "gate_prob_non_ct": gate_prob_non_ct
            }
