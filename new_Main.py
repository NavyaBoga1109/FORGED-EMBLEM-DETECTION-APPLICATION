# import all necessary modules
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import SGD

import numpy as np
import os
from tensorflow.keras.models import load_model
import streamlit as st
import pickle
import numpy as np
# import Main
# model=pickle.load(open('model.pkl','rb'))

import csv

#code to read csv file
def read_csv_to_dict(file_path):
    result = {}
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            result[row[0]] = row[1]
    return result

linkFile = read_csv_to_dict(os.getcwd() + "\FLDS.csv")
# define height and width of the image
HEIGHT = 224
WIDTH = 224
# INIT_LR = 1e-5
INIT_LR = 0.001
EPOCHS = 200
BS = 8
SFACTOR = 26
SFACTOR0 = 60
print("************\n")
print(tf.__version__)
print("************\n")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
# baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(WIDTH, HEIGHT, 3)))

# Load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(WIDTH, HEIGHT, 3))


def build_finetune_model(baseModel, dropout, num_classes):


    # Construct the head of the model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(num_classes, activation="softmax")(headModel)

    # Place the head FC model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)

    # Freeze all layers in the base model
    for layer in baseModel.layers:
        layer.trainable = False

    # Compile the model
    print("[INFO] Compiling model...")
    optimizer = SGD(learning_rate=INIT_LR, momentum=0.9, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    # train the head of the network


# class list of logos
class_list = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds',
              'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo']  # the labels of our data
dropout = 0.5

# model building
model = build_finetune_model(
    baseModel, dropout=dropout, num_classes=len(class_list))
model = load_model('new_logo.model')


# function to prepare image for testing


def predimage(path):
    test = load_img(path, target_size=(HEIGHT, WIDTH))
    test = img_to_array(test)
    test = np.expand_dims(test, axis=0)
    test /= 255
    result = model.predict(test, batch_size=BS)
    result = (result*100)
    result = list(np.around(np.array(result), 1))
    # print(result)
    mx = min(100,np.max(result))
    idx = np.argmax(result)
    return mx,idx



# main function to run the web app & show content at front-end side
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://i.pinimg.com/originals/a7/be/b7/a7beb7f8a8f090b3f7c857543c6e2f72.gif");
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            background-size: 60%;
            background-color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    add_bg_from_url()
    html_temp = """
    <div>
        <h2 style="color:#FFFDFA;text-align:center;font-family:serif;font-style:italic">Forged Emblem Detection Application</h2>
    </div>
    </br>
    </br>
    </br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image = st.file_uploader("Upload Logo Image", type=["png", "jpg", "jpeg"])

    real = """  
      <div style="background-color:#989898;padding:10px  text-align:center;">
       <h2 style="color:#1338BE;text-align:center;font-family:serif"> Logo is <strong>Real<strong> </h2>
       </div>
    """
    fake = """  
      <div style="background-color:#676767;padding:10px; text-align:center;" >
       <h2 style="color:red ;text-align:center;font-family:serif"> Logo is <strong>Fake<strong></h2>
       </div>
    """

    if st.button("Predict"):
        output,idx = predimage(image)
        print(class_list[idx])
        st.success(
            "The probability of logo's originality is {:.2f} %".format(output)
        )

        if SFACTOR0 <= output:
            # print(class_list[int(idx)])
            real = str("<div>") + real + str('<a href = "') + str(linkFile[class_list[int(idx)]]) +str('">Link</a></div>')
            st.markdown(real, unsafe_allow_html=True)
        else:
            st.markdown(fake, unsafe_allow_html=True)


# main function to run the application
if __name__ == '__main__':
    main()
