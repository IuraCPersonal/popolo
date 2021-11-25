from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import torch
import cv2
from torch import nn as nn
import numpy as np
import pandas as pd
import moviepy.editor as mp
from PIL import Image, ImageFilter
from torchvision import transforms, models
from moviepy.editor import VideoFileClip
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['mp4', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_video(filename):
    # Choosing devide to be gpu if have one, else cpu
    DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def get_model():

        model = torchvision.models.resnet101(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        nr_feat = model.fc.in_features
        model.fc = nn.Sequential(OrderedDict(
            [('fc', nn.Linear(nr_feat, 1)), ('sigmoid', nn.Sigmoid())]))

        return model

    import torchvision
    from collections import OrderedDict

    loaded_model = get_model()
    loaded_model.load_state_dict(torch.load(
        r'ai-model/resnet101_classificator.pth', map_location='cpu'), strict=False)

    # load our serialized model and parameter from disk
    modelFile = "ai-model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "ai-model/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    # load the input video and construct an input blob
    # for the video by resizing to a fixed 300x300 pixels
    # and the normalizing it
    # df = pd.DataFrame(columns=['frame', 'x_min', 'y_min', 'x_max', 'y_max'])
    cap = cv2.VideoCapture('static/uploads/' + filename)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('static/uploads/output_kids.mp4', cv2.VideoWriter_fourcc(
        *'DIVX'), fps, (width, height))

    while(True):
        ret, img = cap.read()
        if ret == True:
            # img = cv2.resize(img, None, fx=0.5, fy=0.5)
            height, width = img.shape[:2]
            img1 = img.copy()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # create blob from image
            blob = cv2.dnn.blobFromImage(cv2.resize(
                img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
            # print("First Blob: {}".format(blob.shape))
            # pass the blob through the network and obtain the detections and predictions
            net.setInput(blob)
            faces = net.forward()

            # loop over the detections(faces)
            for i in range(faces.shape[2]):
                # extract the confidence (i.e. probability) associeted with the prediction
                confidence = faces[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is greater than
                # the minimum confidence
                if confidence > 0.5:
                    # compute the (x, y) - coordinates of the bounding box for the object
                    box = faces[0, 0, i, 3:7] * \
                        np.array([width, height, width, height])

                    (x, y, x1, y1) = box.astype("int")
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

                    # new_row = {'frame': current_frame, 'x_min': x, 'y_min': y, 'x_max': x1, 'y_max': y1}
                    # df = df.append(new_row, ignore_index=True)

                    a = (img[y:y1, x:x1])
                # cv2_imshow(a)
                    PIL_image = Image.fromarray(a)
                    tensor = transformations(PIL_image)
                    prediction = loaded_model(tensor.unsqueeze(0))

                    if float(prediction) > .5:
                        # cv2.rectangle(img1, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        face_image = Image.fromarray(img1)
                        cropped_image = face_image.crop((x, y, x1, y1))
                        blurred_image = cropped_image.filter(
                            ImageFilter.GaussianBlur(radius=15))
                        face_image.paste(blurred_image, (x, y, x1, y1))
                        img1 = np.array(face_image)

        # cv2_imshow(img1)
            out.write(img1)

            # cv2_imshow(img1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # print(df)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    cap = VideoFileClip("static/uploads/" + filename)
    audio = cap.audio
    out = VideoFileClip("static/uploads/output_kids.mp4")
    final_video = out.set_audio(audio)
    final_video.write_videofile("static/uploads/final_video_kids.mp4", fps)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        transform_video(filename)
        return render_template('index.html', filename='final_video_kids.mp4')
    else:
        flash('Allowed image types are - mp4')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/final_video_kids.mp4'), code=301)


if __name__ == "__main__":
    app.run()
