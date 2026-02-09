import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.score = None

    def _test_generator(self):
            test_dir = Path(self.config.training_data) / "test"
            
            # CORRECTION: Match the Trainer name
            classes_list = ["normal", "adenocarcinoma"] # <--- CHANGED THIS
            
            datagenerator_kwargs = dict(rescale=1.0 / 255)
            dataflow_kwargs = dict(
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                interpolation="bilinear",
                class_mode="binary",
                shuffle=False, 
                classes=classes_list
            )
            test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
            self.test_generator = test_datagenerator.flow_from_directory(
                directory=str(test_dir), **dataflow_kwargs
            )
    def _build_model(self):
        # Workaround for EfficientNet JSON bug
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
        self.advanced_metrics = self.calculate_advanced_metrics()
        self.save_score()
        self.log_into_mlflow()

    def calculate_advanced_metrics(self):
        self.test_generator.reset()
        y_true = self.test_generator.classes
        y_prob = self.model.predict(self.test_generator)
        y_pred = (y_prob >= 0.5).astype(int).reshape(-1)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5

        cm = confusion_matrix(y_true, y_pred)
        
        print("\n" + "="*40)
        print("📊 CONFUSION MATRIX DEBUG")
        print(f"True Negatives (Normal): {cm[0][0]}")
        print(f"False Positives (Normal->Cancer): {cm[0][1]}")
        print(f"False Negatives (Cancer->Normal): {cm[1][0]}")
        print(f"True Positives (Cancer): {cm[1][1]}")
        print("="*40 + "\n")

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "confusion_matrix": cm
        }

    def save_score(self):
        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1]),
            "precision": self.advanced_metrics["precision"],
            "recall": self.advanced_metrics["recall"],
            "f1_score": self.advanced_metrics["f1_score"],
            "auc_roc": self.advanced_metrics["auc_roc"]
        }
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        # 1. FIX: Connect Tracking URI properly
        username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        
        if username and password:
            mlflow.set_tracking_uri(self.config.mlflow_uri) # <--- ADDED THIS
            mlflow.set_registry_uri(self.config.mlflow_uri)
        else:
            print("⚠️ Authentication failed. Logging locally.")
            mlflow.set_tracking_uri("") # <--- ADDED THIS
            mlflow.set_registry_uri("")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run(run_name="Model_Evaluation"):
            mlflow.log_params(self.config.all_params)
            
            metrics = {
                "loss": self.score[0],
                "accuracy": self.score[1],
                "precision": self.advanced_metrics["precision"],
                "recall": self.advanced_metrics["recall"],
                "f1_score": self.advanced_metrics["f1_score"],
                "auc_roc": self.advanced_metrics["auc_roc"]
            }
            mlflow.log_metrics(metrics)
            
            cm = self.advanced_metrics["confusion_matrix"]
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'Cancer'], 
                        yticklabels=['Normal', 'Cancer'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()
            
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16_Chest_Cancer")
            else:
                mlflow.keras.log_model(self.model, "model")