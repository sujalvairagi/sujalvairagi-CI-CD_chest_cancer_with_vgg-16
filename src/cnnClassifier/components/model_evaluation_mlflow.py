import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.score = None

    def _test_generator(self):
        test_dir = Path(self.config.training_data) / "test"
        datagenerator_kwargs = dict(rescale=1.0 / 255)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="binary",
            shuffle=False
        )
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.test_generator = test_datagenerator.flow_from_directory(
            directory=str(test_dir), **dataflow_kwargs
        )

    def _build_model(self):
        # Rebuild architecture
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=self.config.params_image_size,
            weights=None, include_top=False
        )
        inputs = tf.keras.Input(shape=self.config.params_image_size)
        x = backbone(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def evaluation(self):
        self.model = self._build_model()
        # ✅ FIX: Load weights
        self.model.load_weights(str(self.config.path_of_model))
        
        self._test_generator()
        self.score = self.model.evaluate(self.test_generator)
        self.save_score()

    def save_score(self):
        if self.score is None:
            raise ValueError("Score not found")
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=Path("scores.json"), data=scores)