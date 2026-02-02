import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np


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
        # ✅ Manually rebuild the skeleton to bypass EfficientNet JSON bug
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
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def evaluation(self):
        self.model = self._build_model()
        self.model.load_weights(str(self.config.path_of_model))
        self._test_generator()
        self.score = self.model.evaluate(self.test_generator)
        self.save_score()

    def save_score(self):
        y_true = self.test_generator.classes  # ground truth labels

        y_prob = self.model.predict(self.test_generator)
        y_pred = (y_prob >= 0.5).astype(int).reshape(-1)

        cm = confusion_matrix(y_true, y_pred).tolist()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1]),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm
        }

        save_json(path=Path("scores.json"), data=scores)

    # ✅ THIS IS THE MISSING PIECE
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            
            # Model logging (Conditional based on tracking server)
            mlflow.keras.log_model(self.model, "model")
            run = mlflow.active_run()
                

            tracking_uri = mlflow.get_tracking_uri()

            

            print("\n MLflow Run Link:")
            print(tracking_uri, "\n")



            