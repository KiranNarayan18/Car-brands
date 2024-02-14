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

            model=load_model('model_resnet50.keras')
            x=x/255
            
            x=np.expand_dims(x,axis=0)
            img_data=preprocess_input(x)

            model.predict(img_data)

            np.argmax(model.predict(img_data), axis=1)

            

        except Exception as error:
            logger.error(CustomException(error, sys))

