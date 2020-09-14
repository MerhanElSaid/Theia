import io
import json
import torch 
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from models.gender import loadModel


app = Flask(__name__)
gender_index = {0:"Female", 1:"Male"}
model = loadModel()
model.load_state_dict(torch.load('checkpoints/gender/deploy_80_model.pth.tar')['state_dict'])
model.eval()

def transform_image(image):
    my_transforms =  transforms.Compose([
        transforms.Resize(226),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485],[0.229])
        ])
    return my_transforms(image).unsqueeze(0).repeat(1,3,1,1)

def get_prediction(image):
    tensor = transform_image(image=image)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return gender_index[predicted_idx]

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        img_bytes = data["body"]["formdata"][4]["src"]
        image = Image.open(io.BytesIO(img_bytes))
        gender = get_prediction(image)
        return jsonify({"request_id":"","time_used":0,"faces":[{"face_token":"","face_rectangle":{},"landmark":{},"attributes":{"gender":{"value":gender},"age":{"value":0}}}],"image_id":"","face_num":1})

@app.route('/')
def default():
    return 'Run Ya Habibi'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)