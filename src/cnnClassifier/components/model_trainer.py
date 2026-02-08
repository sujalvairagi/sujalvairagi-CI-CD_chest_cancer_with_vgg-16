import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
import mlflow
import mlflow.keras
import mlflow.tensorflow





# Callback to ensure Graphs appear (Logs metrics every epoch)
class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for key, value in logs.items():
                mlflow.log_metric(key, value, step=epoch)

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def _build_model(self):
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=False
        )
        backbone.trainable = False 
        
        inputs = tf.keras.Input(shape=self.config.params_image_size)
        x = backbone(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_warmup_lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )
        return model

    def get_base_model(self):
        self.model = self._build_model()
        if os.path.exists(self.config.updated_base_model_path):
             self.model.load_weights(str(self.config.updated_base_model_path))

    def train_valid_generator(self):
        train_dir = os.path.join(self.config.training_data, "train")
        val_dir = os.path.join(self.config.training_data, "val")
        
        datagenerator_kwargs = dict(rescale=1.0 / 255)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="binary"
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=val_dir, shuffle=False, **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40, horizontal_flip=True, width_shift_range=0.2,
                height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=train_dir, shuffle=True, **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save_weights(str(path), save_format="h5")

    def train(self):
        # ------------------------------------------------------------------
        # ROBUST CONNECTION: Check for Credentials -> Connect or Fallback
        # ------------------------------------------------------------------
        username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

        if username and password:
            # ✅ Success: Credentials found. Use Remote Dagshub URI.
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            print(f"\n✅ Authentication Successful. Logging metrics to Dagshub: {self.config.mlflow_uri}")
        else:
            # ❌ Failure: Credentials missing. Switch to Local.
            print("\n⚠️  WARNING: MLFLOW_TRACKING_USERNAME or PASSWORD not found in OS environment.")
            print("⚠️  Authentication Failed. Switching to LOCAL tracking.")
            print("    -> Graphs will be saved locally in the './mlruns' folder.")
            mlflow.set_tracking_uri("") # Empty string forces local file storage
        
        mlflow.set_tracking_uri(self.config.mlflow_uri)

        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        custom_callback = MLflowLoggingCallback()

        # ============================================================
        # RUN 1: WARMUP (Head Training Only)
        # ============================================================
        print("\n========== STAGE 1: Head Training ==========")
        with mlflow.start_run(run_name="Stage-1_Head_Training"):
            # 1. LOG PARAMS (FIXED)
            mlflow.log_params({
                "epochs": self.config.params_warmup_epochs,
                "learning_rate": self.config.params_warmup_lr,
                "batch_size": self.config.params_batch_size,
                "phase": "Warmup (Head Only)"
            })
            
            self.model.fit(
                self.train_generator,
                epochs=self.config.params_warmup_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.valid_generator,
                validation_steps=validation_steps,
                callbacks=[custom_callback]
            )
            self.save_model(path=Path("artifacts/training/model_head_only.h5"), model=self.model)

        # ============================================================
        # RUN 2: FINE-TUNING (Accuracy Boosting)
        # ============================================================
        print("\n========== STAGE 2: Fine-Tuning ==========")
        
        self.model.trainable = True
        fine_tune_at = len(self.model.layers) - self.config.params_fine_tune_layers
        
        for index, layer in enumerate(self.model.layers):
            if index < fine_tune_at:
                layer.trainable = False
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_fine_tune_lr, momentum=0.9),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
        )

        with mlflow.start_run(run_name="Stage-2_Fine_Tuning"):
            # 1. LOG PARAMS (FIXED)
            mlflow.log_params({
                "epochs": self.config.params_fine_tune_epochs,
                "learning_rate": self.config.params_fine_tune_lr,
                "fine_tune_layers": self.config.params_fine_tune_layers,
                "phase": "Fine-Tuning"
            })

            self.model.fit(
                self.train_generator,
                epochs=self.config.params_fine_tune_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.valid_generator,
                validation_steps=validation_steps,
                callbacks=[custom_callback, early_stopping, reduce_lr]
            )

        self.save_model(path=self.config.trained_model_path, model=self.model)