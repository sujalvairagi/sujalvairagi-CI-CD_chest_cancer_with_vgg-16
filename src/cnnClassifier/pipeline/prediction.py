import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.cancer_model = self._load_cancer_model()
        self.gate_model = self._load_gate_model()

    def _load_cancer_model(self):
        # Reconstruct structure to match weights
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3), weights=None, include_top=False
        )
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = backbone(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        model = tf.keras.Model(inputs, outputs)
        model_path = os.path.join("artifacts", "training", "model.h5")
        
        try:
            model.load_weights(model_path)
        except:
            model = tf.keras.models.load_model(model_path)
        return model

    def _load_gate_model(self):
        return tf.keras.models.load_model(
            os.path.join("artifacts", "ct_gate", "ct_gate_model.h5"), 
            compile=False
        )

    def _preprocess(self):
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict(self):
        img_array = self._preprocess()

        # --- 1. CT Gate Check ---
        # Output 0 = CT, 1 = Non-CT
        gate_prob_non_ct = float(self.gate_model.predict(img_array)[0][0])

        # If strict "Not a CT" (Probability > 50%)
        if gate_prob_non_ct > 0.5:
            return {
                "is_ct": False,
                "label": "Not a CT Scan",
                "probability": gate_prob_non_ct,
                "gate_p": gate_prob_non_ct,   # RAW VALUE
                "cancer_p": None              # Didn't run
            }

        # --- 2. Cancer Diagnosis (Only runs if it IS a CT) ---
        pred = self.cancer_model.predict(img_array)
        p_cancer = float(pred[0][0])  # 0.0 (Normal) to 1.0 (Cancer)

        # STRICT Binary Classification (Threshold 0.5)
        if p_cancer > 0.5:
            final_label = "Adenocarcinoma"
            final_prob = p_cancer
        else:
            final_label = "Normal"
            final_prob = 1 - p_cancer 

        if 0.30 <= p_cancer <= 0.70:
            final_label += " (with slight uncertainty)"

        return {
            "is_ct": True,
            "label": final_label,
            "probability": final_prob,
            "gate_p": gate_prob_non_ct, # RAW VALUE
            "cancer_p": p_cancer        # RAW VALUE
        }