import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # ✅ Load the model ONCE during initialization
        self.model = self._load_model()

    def _load_model(self):
        # Build architecture
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
        model = tf.keras.Model(inputs, outputs)
        
        # Load weights
        model_path = os.path.join("artifacts", "training", "model.h5")
        model.load_weights(model_path)
        return model

    def predict(self):
        # Preprocess image
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # ✅ Use the pre-loaded model
        prediction = self.model.predict(img_array)
        
        if prediction[0][0] > 0.5:
            return [{"image": "Normal"}]
        else:
            return [{"image": "Adenocarcinoma Cancer"}]