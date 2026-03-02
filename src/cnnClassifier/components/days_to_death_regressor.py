import os
import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import DaysToDeathConfig
import mlflow
import mlflow.keras
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.ERROR)

_SHUFFLE_SEED = 42


class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for key, value in logs.items():
                mlflow.log_metric(key, value, step=epoch)


class DaysToDeathRegressor:
    """
    Visual biomarker regression model that predicts days-to-death
    from chest CT scan images using an EfficientNetB0 backbone.

    Expected data layout under config.data_dir:
        <data_dir>/
            train/
                images/   (chest CT PNG/JPEG files)
                labels.csv  (columns: filename, days_to_death)
            val/
                images/
                labels.csv
    """

    def __init__(self, config: DaysToDeathConfig):
        self.config = config
        self._model = None  # cached model for inference

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    def _build_model(self) -> tf.keras.Model:
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=self.config.params_image_size,
            weights="imagenet",
            include_top=False
        )
        backbone.trainable = False

        inputs = tf.keras.Input(shape=self.config.params_image_size)
        x = backbone(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(self.config.params_dropout_rate)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.config.params_dropout_rate)(x)
        # Linear output — predicts a continuous days-to-death value
        outputs = tf.keras.layers.Dense(1, activation="linear", name="days_to_death")(x)

        model = tf.keras.Model(inputs, outputs, name="DaysToDeathRegressor")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
            ]
        )
        return model

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_dataset_from_csv(
        images_dir: str,
        labels_csv: str,
        image_size: tuple,
        batch_size: int,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Builds a tf.data.Dataset from a CSV file that maps
        image filenames to days-to-death float labels.

        CSV format (with header):
            filename,days_to_death
            scan001.jpg,342
            scan002.jpg,87
            ...
        """
        filenames, labels = [], []
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filenames.append(os.path.join(images_dir, row["filename"]))
                labels.append(float(row["days_to_death"]))

        filenames = tf.constant(filenames)
        labels = tf.constant(labels, dtype=tf.float32)

        h, w = image_size[0], image_size[1]

        def load_image(path, label):
            raw = tf.io.read_file(path)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img = tf.image.resize(img, [h, w])
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(filenames), seed=_SHUFFLE_SEED)
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------
    def train(self):
        # MLflow authentication
        username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        if username and password:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            print(f"\n✅ Auth successful. Logging to Dagshub: {self.config.mlflow_uri}")
        else:
            print("\n⚠️  MLflow credentials not found. Switching to local tracking.")
            mlflow.set_tracking_uri("")

        train_images_dir = os.path.join(self.config.data_dir, "train", "images")
        train_labels_csv = os.path.join(self.config.data_dir, "train", "labels.csv")
        val_images_dir = os.path.join(self.config.data_dir, "val", "images")
        val_labels_csv = os.path.join(self.config.data_dir, "val", "labels.csv")

        train_ds = self._load_dataset_from_csv(
            images_dir=train_images_dir,
            labels_csv=train_labels_csv,
            image_size=self.config.params_image_size,
            batch_size=self.config.params_batch_size,
            shuffle=True
        )
        val_ds = self._load_dataset_from_csv(
            images_dir=val_images_dir,
            labels_csv=val_labels_csv,
            image_size=self.config.params_image_size,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

        model = self._build_model()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=5,
            restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        mlflow_callback = MLflowLoggingCallback()

        with mlflow.start_run(run_name="DaysToDeathRegression_Training"):
            mlflow.log_params({
                "epochs": self.config.params_epochs,
                "batch_size": self.config.params_batch_size,
                "learning_rate": self.config.params_learning_rate,
                "dropout_rate": self.config.params_dropout_rate,
                "model_type": "EfficientNetB0_DaysToDeathRegressor",
                "loss": "MAE",
            })

            print("\n========== Days-to-Death Regression Training ==========")
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.params_epochs,
                callbacks=[early_stopping, reduce_lr, mlflow_callback]
            )

        os.makedirs(os.path.dirname(str(self.config.model_path)), exist_ok=True)
        model.save(str(self.config.model_path))

        deployment_path = os.path.join("model", "days_to_death_model.h5")
        os.makedirs("model", exist_ok=True)
        model.save(deployment_path)

        print(f"\n✅ Days-to-death model saved: {deployment_path}")

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------
    def predict_days_to_death(self, image_path: str) -> float:
        """
        Predict days-to-death for a single image.
        The model is loaded once and cached for subsequent calls.

        Args:
            image_path: Path to the chest CT image file.

        Returns:
            Predicted number of days to death (float).
        """
        if self._model is None:
            self._model = tf.keras.models.load_model(str(self.config.model_path))

        h, w = self.config.params_image_size[0], self.config.params_image_size[1]

        raw = tf.io.read_file(image_path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [h, w])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)

        prediction = self._model.predict(img, verbose=0)
        return float(prediction[0][0])
