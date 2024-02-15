from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image

from prediction import Prediction


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file:
            # Load image
            image = Image.open(file)

            # img =image.load_img(file,target_size=(224,224))

            open_cv_image = np.array(image)

            model_obj = Prediction()
            predictions = model_obj.predict(open_cv_image)

            # Render template with predictions
            return render_template('results.html', predictions=predictions)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
