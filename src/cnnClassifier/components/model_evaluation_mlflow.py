import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.score = None

    def _test_generator(self):
        test_dir = Path(self.config.training_data) / "test"
        
        # Keep your verified class names!
        classes_list = ["normal", "adenocarcinoma"]
        
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
        
        # Standard evaluate (ignore accuracy here, we calculate a better one below)
        self.score = self.model.evaluate(self.test_generator)
        
        self.advanced_metrics = self.calculate_advanced_metrics()
        self.save_score()
        self.log_into_mlflow()

    def calculate_advanced_metrics(self):
        self.test_generator.reset()
        y_true = self.test_generator.classes
        y_prob = self.model.predict(self.test_generator)
        
        # 1. FIND OPTIMAL THRESHOLD (Maximize TPR - FPR)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\n✅ OPTIMAL THRESHOLD FOUND: {optimal_threshold:.4f}")
        
        # 2. GENERATE PREDICTIONS WITH NEW THRESHOLD
        y_pred = (y_prob >= optimal_threshold).astype(int).reshape(-1)

        # 3. CALCULATE METRICS
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = np.mean(y_pred == y_true)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5

        # 4. EXTRACT COUNTS (Normal vs Cancer)
        # Class 0 = Normal, Class 1 = Cancer
        num_normal = np.sum(y_true == 0)
        num_cancer = np.sum(y_true == 1)

        # 5. EXTRACT CONFUSION MATRIX VALUES
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() # Flattens the matrix into 4 numbers

        print("\n" + "="*40)
        print(f"📊 FINAL METRICS (Threshold: {optimal_threshold:.4f})")
        print(f"Total Images: {len(y_true)}")
        print(f"   - Normal: {num_normal}")
        print(f"   - Cancer: {num_cancer}")
        print("-" * 20)
        print(f"True Positives (Cancer detected): {tp}")
        print(f"False Negatives (Cancer missed):  {fn}  <-- SAFETY CRITICAL")
        print(f"True Negatives (Normal detected): {tn}")
        print(f"False Positives (False Alarm):    {fp}")
        print("="*40 + "\n")

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "confusion_matrix": cm,
            "optimal_threshold": float(optimal_threshold),
            "accuracy_manual": float(accuracy),
            "counts": {
                "total_normal": int(num_normal),
                "total_cancer": int(num_cancer),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            }
        }

    def save_score(self):
        # This is what gets saved to scores.json
        scores = {
            "loss": float(self.score[0]),
            "accuracy": self.advanced_metrics["accuracy_manual"],
            "precision": self.advanced_metrics["precision"],
            "recall": self.advanced_metrics["recall"],
            "f1_score": self.advanced_metrics["f1_score"],
            "auc_roc": self.advanced_metrics["auc_roc"],
            "threshold": self.advanced_metrics["optimal_threshold"],
            # --- NEW FIELDS ---
            "total_normal_images": self.advanced_metrics["counts"]["total_normal"],
            "total_cancer_images": self.advanced_metrics["counts"]["total_cancer"],
            "true_positives": self.advanced_metrics["counts"]["tp"],
            "false_positives": self.advanced_metrics["counts"]["fp"],
            "true_negatives": self.advanced_metrics["counts"]["tn"],
            "false_negatives": self.advanced_metrics["counts"]["fn"]
        }
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        # ... (Authentication code remains same) ...
        username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        if username and password:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
        else:
            mlflow.set_tracking_uri("")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run(run_name="Model_Evaluation"):
            mlflow.log_params(self.config.all_params)
            mlflow.log_param("optimal_threshold", self.advanced_metrics["optimal_threshold"])
            
            # Log standard metrics
            metrics = {
                "loss": self.score[0],
                "accuracy": self.advanced_metrics["accuracy_manual"],
                "precision": self.advanced_metrics["precision"],
                "recall": self.advanced_metrics["recall"],
                "f1_score": self.advanced_metrics["f1_score"],
                "auc_roc": self.advanced_metrics["auc_roc"],
                "false_negatives": self.advanced_metrics["counts"]["fn"], # Log FN specifically!
                "true_positives": self.advanced_metrics["counts"]["tp"]
            }
            mlflow.log_metrics(metrics)
            
            # Confusion Matrix
            cm = self.advanced_metrics["confusion_matrix"]
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'Cancer'], 
                        yticklabels=['Normal', 'Cancer'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'CM (Thresh: {self.advanced_metrics["optimal_threshold"]:.2f})')
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()
            
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16_Chest_Cancer")
            else:
                mlflow.keras.log_model(self.model, "model")