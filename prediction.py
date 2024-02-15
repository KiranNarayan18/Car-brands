import cv2
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.logger import logger, CustomException

class Prediction:
    def __init__(self):
        pass

    def predict(self,img):
        try:
            
            resize = cv2.resize(img, (224,224))
            x=image.img_to_array(resize)

            x=x/255

            x=np.expand_dims(x,axis=0)
            img_data=preprocess_input(x)
            
            model=load_model('models\model_resnet50.keras')
            model.predict(img_data)

            a=np.argmax(model.predict(img_data), axis=1)

            return a


        except Exception as error:
            logger.error(CustomException(error, sys))

