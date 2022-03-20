from flask import Flask, render_template, request
import pickle
import cv2 as cv
import numpy as np
import base64
from io import BytesIO
from PIL import Image

model = pickle.load(open('mnist.pkl','rb'))

app = Flask(__name__,template_folder='template')


@app.route('/')
def main():
    return render_template('index.html')

# def predict_img(img):
#     img_3d=img.reshape(-1,28,28)
#     im_resize=img_3d/225.0
#     prediction=model.predict(im_resize)
#     pred=np.argmax(prediction)
#     return pred

@app.route('/predict', methods = ['POST'])
def predict():
    img = request.form.get('image').replace('data:image/png;base64,','')
    img = img.encode()
    print(img)
    print(type(img))

    with open("image.png","wb") as fh:
        fh.write(base64.decodebytes(img))
    
    im = Image.open('image.png').convert('RGBA')
    bg = Image.new('RGBA', im.size, (255,255,255))

    alpha = Image.alpha_composite(bg,im)
    # alpha = im
    alpha = alpha.convert('L')
    alpha = alpha.resize((28,28))


    image_np = np.array(alpha).reshape(784,)

    lst = []
    for i in image_np:
        if i==255:
            lst.append(0)
        else:
            lst.append(i)
    

    print(lst)
    # print(image_np)
    print(image_np.shape)


    prediction = model.predict([image_np])
    return str(prediction)
if __name__== '__main__':

    app.run(host= '0.0.0.0')
    app.run(debug=True)