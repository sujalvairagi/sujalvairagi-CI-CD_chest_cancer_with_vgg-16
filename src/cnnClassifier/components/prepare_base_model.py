import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.EfficientNetB0(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=False
        )

    @staticmethod
    def _prepare_full_model(model, learning_rate):
        model.trainable = False
        inputs = tf.keras.Input(shape=model.input_shape[1:])
        x = model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        full_model = tf.keras.Model(inputs, outputs)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )
        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # ✅ FIX: Explicitly save as HDF5 weights
        model.save_weights(str(path), save_format="h5")