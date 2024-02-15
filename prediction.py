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
            
            x = cv2.resize(img, (224,224))
           
            img_data = x.reshape(1,224,224,3)
            x = x / 255
            
            model=load_model('models\model_resnet50.keras')
            model.predict(img_data)

            a=np.argmax(model.predict(img_data), axis=1)

            return a


        except Exception as error:
            logger.error(CustomException(error, sys))


if __name__ == "__main__":
    
    path = 'datasets/Test/lamborghini/10.jpg'

    img =image.load_img(path,target_size=(224,224))
  
    obj = Prediction()
    print(obj.predict(img))
