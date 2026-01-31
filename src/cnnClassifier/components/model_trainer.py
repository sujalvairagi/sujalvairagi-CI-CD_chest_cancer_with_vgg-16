import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def _build_model(self):
        # Rebuild architecture manually
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
        
        # Compile needed for training
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_warmup_lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )
        return model

    def get_base_model(self):
        self.model = self._build_model()
        # ✅ FIX: Load weights from the H5 file
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
            directory=val_dir,
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=train_dir,
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # ✅ FIX: Save trained weights as H5
        model.save_weights(str(path), save_format="h5")

    def train(self):
        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Phase 1: Warmup
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_warmup_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=validation_steps
        )

        # Phase 2: Fine-tune
        # Unfreeze layers if needed (simplified for stability)
        self.model.trainable = True
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_fine_tune_lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_fine_tune_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=validation_steps
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)