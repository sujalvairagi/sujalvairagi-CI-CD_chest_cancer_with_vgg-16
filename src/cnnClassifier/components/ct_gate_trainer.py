import os
import tensorflow as tf
from cnnClassifier.entity.config_entity import CTGateConfig
import mlflow
import mlflow.keras



class CTGateTrainer:
    def __init__(self, config: CTGateConfig):
        self.config = config

    def _build_model(self):
        inputs = tf.keras.Input(shape=self.config.params_image_size)

        # --- BRANCH 1: grayscale math ---
        gray_score = tf.keras.layers.Lambda(
            lambda x: tf.math.reduce_std(x, axis=-1, keepdims=True),
            name="grayscale_math_layer"
        )(inputs)
        gray_score = tf.keras.layers.GlobalAveragePooling2D()(gray_score)

        # --- BRANCH 2: CNN texture ---
        x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        cnn_feat = tf.keras.layers.GlobalAveragePooling2D()(x)

        # --- FUSION ---
        combined = tf.keras.layers.Concatenate()([gray_score, cnn_feat])

        x = tf.keras.layers.Dense(64, activation="relu")(combined)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CT_Gate_Model")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                "accuracy",
                tf.keras.metrics.FalsePositives(name="false_positives"),
                tf.keras.metrics.Precision(name="precision")
            ]
        )
        return model

    def train(self):
        train_dir = os.path.join(self.config.data_dir, "train")
        val_dir = os.path.join(self.config.data_dir, "val")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=train_dir,
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            label_mode="binary",
            shuffle=True
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=val_dir,
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            label_mode="binary",
            shuffle=False
        )

        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        model = self._build_model()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
        mlflow.keras.autolog()

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.params_epochs,
            callbacks=[early_stopping]
        )

        model.save(str(self.config.model_path))
