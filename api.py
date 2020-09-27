import io
import json
import torch 
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from pathlib import Path
import urllib.request

from models.gender import loadGenderModel
from models.age import loadAgeModel
from defaults import _C as cfg

app = Flask(__name__)

#### Gender Model
gender_index = {0:"Female", 1:"Male"}
gender_model = loadGenderModel()
gender_model.load_state_dict(torch.load('checkpoints/gender/deploy_80_model.pth.tar')['state_dict'])
gender_model.eval()

### Age Model
age_model = loadAgeModel(model_name=cfg.MODEL.ARCH, pretrained=None)
device = "cuda" if torch.cuda.is_available() else "cpu"
age_model = age_model.to(device)

# load checkpoint
resume_path = Path(__file__).resolve().parent.joinpath("checkpoints/age", "epoch044_0.02343_3.9984.pth")

if not resume_path.is_file():
    print(f"=> model path is not set; start downloading trained model to {resume_path}")
    url = "https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"
    urllib.request.urlretrieve(url, str(resume_path))
    print("=> download finished")

if Path(resume_path).is_file():
    print("=> loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path, map_location="cpu")
    age_model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(resume_path))
else:
    raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

if device == "cuda":
    cudnn.benchmark = True

age_model.eval()

def transform_image(image):
    my_transforms =  transforms.Compose([
        transforms.Resize(226),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485],[0.229])
        ])
    return my_transforms(image).unsqueeze(0).repeat(1,3,1,1)

def transform_age_image(image):
    img_size = 224
    faces = np.empty((1, img_size, img_size, 3))
    faces[0] = image.resize((img_size,img_size))
    inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to('cuda')
    return inputs

def get_gender_prediction(image):
    tensor = transform_image(image=image)
    outputs = gender_model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return gender_index[predicted_idx]


def get_age_prediction(image):
    with torch.no_grad():
        tensor = transform_age_image(image=image)
        outputs = F.softmax(age_model(tensor), dim=-1).cpu().numpy()
        ages = np.arange(0, 101)
        predicted_ages = (outputs * ages).sum(axis=-1)[0]
        return predicted_ages

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        img_bytes = request.files['image_file'].read()
        image = Image.open(io.BytesIO(img_bytes))
        gender = get_gender_prediction(image)
        age = get_age_prediction(image)
        return jsonify({"request_id":"","time_used":0,"faces":[{"face_token":"","face_rectangle":{},"landmark":{},"attributes":{"gender":{"value":gender},"age":{"value":age}}}],"image_id":"","face_num":1})

@app.route('/')
def default():
    return 'API Working'

if __name__ == '__main__':
    app.run()