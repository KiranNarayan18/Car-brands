import os
import sys
import glob
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda 



from src.logger import logger, CustomException
from src.config.configuration import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config = config
        self.train_path = f"{self.config.model_dir}/Train"
        self.test_path = f"{self.config.model_dir}/Test"

    def train(self):
        try:
            IMAGE_SIZE = [224, 244]

            model = ResNet50(inpput_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

            for layer in model.layers:
                layer.trainable = False

            folders = glob(self.train_path)
            
            x = Flatten()(model.output)
            predictions = Dense(len(folders), activation='softmax')(x)
            model = Model(inputs=model.inputs, outputs=predictions)

            print("Model Summary")
            print(model.summary())

            model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                    )
            

            train_datagen = ImageGenerator(rescale = 1./ 255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)

            test_datagen = ImageGenerator(rescale = 1./255)


            train_dataset = train_datagen.flow_from_directory(
                self.config.train_path,
                target_size=IMAGE_SIZE,
                batch_size=32,
                class_mode='categorical'

            )


            test_dataset = test_datagen.flow_from_directory(
                self.config.test_path,
                target_size=IMAGE_SIZE,
                batch_size=32,
                class_mode='categorical'
            )


            r = model.fit(
                train_dataset,
                validation_dataset=test_dataset,
                epochs=50,
                steps_per_epoch=len(train_dataset),
                validation_steps=len(test_dataset)                
            )


            plt.plot(r.history['loss'], label='train loss')
            plt.plot(r.history['val_loss'], label='val loss')
            plt.legend()
            plt.savefig('LossVal_loss')


            plt.plot(r.history['accuracy'], label='train acc')
            plt.plot(r.history['val_accuracy'], label='val acc')
            plt.legend()
            plt.savefig('AccVal_acc')


            model.save(f'{self.config.model_dir}/model_resnet50.h5')

        except Exception as error:
            logger.error(CustomException(error, sys))