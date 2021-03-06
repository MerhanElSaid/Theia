import io
import json
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import os
import urllib.request
from flask import Flask, jsonify, request
import random
import cv2
from models.gender import Model_gend
from models.age import loadAgeModel
from models.Facial_Exp import Face_Emotion_CNN

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
#### Gender Model

gender_index = {1: "Female", 0: "Male"}
gender_model = Model_gend()
gender_model.load_state_dict(torch.load('checkpoints/gender/Enhanced_Gen_colored.pth', map_location=torch.device(device)))
gender_model.eval()

### Age Model
Age_classes = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3,
           30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5,
           50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7,
           70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9,
           90, 91, 92, 93]
age_model = loadAgeModel()
if not os.path.isfile("checkpoints/age/colored_model_res50_.pth"):
    urllib.request.urlretrieve("https://drive.google.com/uc?export=download&id=1bn0zN6bmf5kBU0p0LghbtVjbGyy0uTyQ", "checkpoints/age/colored_model_res50_.pth")
age_model.load_state_dict(torch.load('checkpoints/age/colored_model_res50_.pth', map_location=torch.device(device)))

age_model.eval()

#"https://drive.google.com/file/d/1VjEEjOfOkFHc0Qb4rihX-2SEPac41xah/view?usp=sharing"
# load checkpoint
#resume_path = Path(__file__).resolve().parent.joinpath("checkpoints/age", "79.pth")

### Facial Expression Model

FER_2013_EMO_DICT = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness', 4: 'Anger', 5: 'Disgust', 6: 'Fear'}
Exp_model = Face_Emotion_CNN()
Exp_model.load_state_dict(torch.load('checkpoints/Facial_Exp/Model5.pth', map_location=torch.device(device)))

Exp_model.eval()


if device == "cuda":
    cudnn.benchmark = True


def transform_gender_image(image):
    my_transforms = transforms.Compose([transforms.Resize(50),
                                        transforms.RandomCrop(48),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    return my_transforms(image).unsqueeze(0)


def transform_age_image(image):
    age_trans = transforms.Compose([transforms.Resize(130),
                                         transforms.RandomCrop(128),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return age_trans(image).unsqueeze(0)


def transform_facial_exp(image):
    exp_trans = transforms.Compose([transforms.Resize(50),
                                    transforms.CenterCrop(48),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485],
                                                         [0.229])])
    return exp_trans(image).unsqueeze(0)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('scripts/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        flag = 0
        faces = [0]
    else:
        flag = 1
        (x, y, w, h) = faces[0]

    return flag, faces[0]


def image_crop(img, x, y, w, h):
    _, _, ch = img.shape
    cropped = np.zeros((w, h, ch))
    for i in range(ch):
        cropped[:, :, i] = img[y:y + w, x:x + h, i]

    return cropped


def get_gender_prediction(image):
    im = np.array(image)
    flag, face = detect_face(im)

    if flag == 1:
        image = image_crop(im, face[0], face[1], face[2], face[3])
        image = Image.fromarray(np.uint8(image))

    tensor = transform_gender_image(image=image)
    tensor = tensor.to(device)
    outputs = gender_model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return gender_index[predicted_idx]


def get_age_prediction(image):
    im = np.array(image)
    flag, face = detect_face(im)

    if flag == 1:
        image = image_crop(im, face[0], face[1], face[2], face[3])
        image = Image.fromarray(np.uint8(image))

    with torch.no_grad():
        tensor = transform_age_image(image=image)
        tensor = tensor.to(device)
        outputs = age_model(tensor)
        _, pred = torch.topk(outputs, 1)
        pred = Age_classes[pred]
        return int(pred)



def get_expr_prediction(image):
    im = np.array(image)
    flag, face = detect_face(im)

    if flag == 1:
        image = image_crop(im, face[0], face[1], face[2], face[3])
        image = Image.fromarray(np.uint8(image))

    with torch.no_grad():
        image = transform_facial_exp(image)
        image = image.to(device)
        output = Exp_model(image)
        prob = torch.softmax(output, 1)[0]
        _, prediction = torch.topk(prob, 1)
        mood_items = {}
        for i in range(0, len(FER_2013_EMO_DICT)):
            emotion_label = FER_2013_EMO_DICT[i]
            emotion_prediction = 100 * prob[i].item()
            mood_items[emotion_label] = emotion_prediction
        emotion = FER_2013_EMO_DICT[int(prediction.squeeze().item())]
        return mood_items, emotion


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_bytes = request.files['image_file'].read()
        image = Image.open(io.BytesIO(img_bytes))
        gender = get_gender_prediction(image)
        age = get_age_prediction(image)
        expression, most_prob = get_expr_prediction(image)
        return jsonify({"request_id": "", "time_used": 0,
                        "faces": [{"face_token": "", "face_rectangle": {}, "landmark": {},
                                   "attributes": {"gender": {"value": gender}, "age": {"value": age},
                                                  "expressions": [{"value": expression}, {"probably": most_prob}]}}],
                        "image_id": "",
                        "face_num": 1})

@app.route('/')
def default():
    return 'API Working'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
