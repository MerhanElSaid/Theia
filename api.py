import io
import json
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from flask import Flask, jsonify, request
import urllib.request

from models.gender import loadGenderModel
from models.age import loadAgeModel
from models.Facial_Exp import Face_Emotion_CNN

app = Flask(__name__)

#### Gender Model

gender_index = {0: "Female", 1: "Male"}
gender_model = loadGenderModel()
gender_model.load_state_dict(torch.load('checkpoints/gender/deploy_80_model.pth.tar', map_location=torch.device('cpu'))['state_dict'])
gender_model.eval()

### Age Model
age_model = loadAgeModel(model_name="se_resnext50_32x4d", pretrained=None)
device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.isfile('checkpoints/age/79.pth'):
    urllib.request.urlretrieve("https://drive.google.com/uc?export=download&id=19UZB5VZvaQZSltXWeNWYo2v5W892uQ9G", "checkpoints/age/79.pth")
checkpoint = torch.load('checkpoints/age/79.pth', map_location="cpu")
age_model.load_state_dict(checkpoint['state_dict'])
age_model = age_model.to(device)
age_model.eval()

# load checkpoint
#resume_path = Path(__file__).resolve().parent.joinpath("checkpoints/age", "79.pth")

### Facial Expression Model

FER_2013_EMO_DICT = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness', 4: 'Anger', 5: 'Disgust', 6: 'Fear'}
Exp_model = Face_Emotion_CNN()
Exp_model.load_state_dict(torch.load('checkpoints/Facial_Exp/FER_trained_model.pt', map_location="cpu"))
Exp_model.eval()


if device == "cuda":
    cudnn.benchmark = True


def transform_gender_image(image):
    my_transforms = transforms.Compose([
        transforms.Resize(226),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    return my_transforms(image).unsqueeze(0).repeat(1, 3, 1, 1)


def transform_age_image(image):
    exp_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    return exp_trans(image).unsqueeze(0).repeat(1, 3, 1, 1)


def transform_facial_exp(image):
    exp_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])
    return exp_trans(image).unsqueeze(0)


def get_gender_prediction(image):
    tensor = transform_gender_image(image=image)
    outputs = gender_model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return gender_index[predicted_idx]


def get_age_prediction(image):
    with torch.no_grad():
        tensor = transform_age_image(image=image)
        outputs = age_model(tensor)
        _, pred = torch.topk(outputs, 1)
        age_vector = pred * 4
        predicted_ages = int(age_vector)
        return int(predicted_ages)


def get_expr_prediction(image):
    with torch.no_grad():
        image = transform_facial_exp(image)
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
    app.run()
