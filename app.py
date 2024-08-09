from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
import pandas as pd
import gc

app = Flask(__name__)

# Load the trained model
model = load_model('static/classifier.h5')

# Load class names
def load_class_names():
    labels = pd.read_csv('static/labels.csv')  # Make sure you have labels.csv in static
    classes = sorted(list(set(labels['breed'])))
    return classes

classes = load_class_names()

# Preprocess image
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Define the extact_features function if not already defined
def get_features(model_name, model_preprocessor, input_size, data):
    from keras.layers import Input, Lambda, GlobalAveragePooling2D
    from keras.models import Model
    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False, input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)
    
    feature_maps = feature_extractor.predict(data, verbose=1)
    return feature_maps

def extact_features(data):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocessor
    from keras.applications.xception import Xception, preprocess_input as xception_preprocessor
    from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inc_resnet_preprocessor
    from keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_preprocessor

    inception_features = get_features(InceptionV3, inception_preprocessor, (331, 331, 3), data)
    xception_features = get_features(Xception, xception_preprocessor, (331, 331, 3), data)
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, (331, 331, 3), data)
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, (331, 331, 3), data)

    final_features = np.concatenate([inception_features, xception_features, nasnet_features, inc_resnet_features], axis=-1)
    
    del inception_features, xception_features, nasnet_features, inc_resnet_features
    gc.collect()
    
    return final_features

# Predict breed
def predict_breed(image_path):
    img = preprocess_image(image_path, target_size=(331, 331, 3))
    features = extact_features(img)
    pred = model.predict(features)
    predicted_class = classes[np.argmax(pred[0])]
    probability = np.max(pred[0])
    return predicted_class, probability

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            breed, prob = predict_breed(file_path)
            os.remove(file_path)  # Remove the file after prediction
            return render_template('result.html', breed=breed, prob=prob)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

'''import os
import numpy as np
import pandas as pd
import gc
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = load_model('static/classifier.h5')

# Load class names
def load_class_names():
    labels = pd.read_csv('static/labels.csv')
    classes = sorted(list(set(labels['breed'])))
    return classes

classes = load_class_names()

# Preprocess image
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Define the extract_features function
def get_features(model_name, model_preprocessor, input_size, data):
    from keras.layers import Input, Lambda, GlobalAveragePooling2D
    from keras.models import Model
    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False, input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    feature_maps = feature_extractor.predict(data, verbose=1)
    return feature_maps

def extract_features(data):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocessor
    from keras.applications.xception import Xception, preprocess_input as xception_preprocessor
    from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inc_resnet_preprocessor
    from keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_preprocessor

    inception_features = get_features(InceptionV3, inception_preprocessor, (331, 331, 3), data)
    xception_features = get_features(Xception, xception_preprocessor, (331, 331, 3), data)
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, (331, 331, 3), data)
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, (331, 331, 3), data)

    final_features = np.concatenate([inception_features, xception_features, nasnet_features, inc_resnet_features], axis=-1)

    del inception_features, xception_features, nasnet_features, inc_resnet_features
    gc.collect()

    return final_features

# Predict breed
def predict_breed(image_path):
    img = preprocess_image(image_path, target_size=(331, 331, 3))
    features = extract_features(img)
    pred = model.predict(features)
    predicted_class = classes[np.argmax(pred[0])]
    probability = np.max(pred[0])
    return predicted_class, probability

@app.post("/uploadfile/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    try:
        file_location = f"static/{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        breed, prob = predict_breed(file_location)
        os.remove(file_location)

        return templates.TemplateResponse("result.html", {"request": request, "breed": breed, "prob": prob})
    except Exception as e:
        print(f"Error: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "error_message": str(e)})

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)'''
