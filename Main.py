# import all necessary modules
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import os
from tensorflow.keras.models import load_model
import streamlit as st
import pickle
import numpy as np
# import Main
# model=pickle.load(open('model.pkl','rb'))

# define height and width of the image
HEIGHT = 224
WIDTH = 224
INIT_LR = 1e-5
EPOCHS = 200
BS = 8
SFACTOR = 99.8

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(WIDTH, HEIGHT, 3)))


def build_finetune_model(baseModel, dropout, num_classes):
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(dropout)(headModel)
    headModel = Dense(num_classes, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    # compile our model
    # print("[INFO] compiling model...")
    #sgd = SGD(lr=INIT_LR,momentum=0.9,nesterov=False)
    model.compile(loss="categorical_crossentropy",optimizer='sgd', metrics=["accuracy"])
    # train the head of the network


# class list of logos
class_list = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds',
              'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo']  # the labels of our data
dropout = 0.5

# model building
model = build_finetune_model(
    baseModel, dropout=dropout, num_classes=len(class_list))
model = load_model('logo.model')


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
    return mx



# main function to run the web app & show content at front-end side
def main():
    st.title("Fake Logo Detection")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Fake Logo Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image = st.file_uploader("Upload Logo Image", type=["png", "jpg", "jpeg"])

    real = """  
      <div style="background-color:black;padding:10px  text-align:center;">
       <h2 style="color:white;text-align:center;"> Logo is <strong>Real<strong> </h2>
       </div>
    """
    fake = """  
      <div style="background-color:black;padding:10px; text-align:center;" >
       <h2 style="color:white ;text-align:center;"> Logo is <strong>Fake<strong></h2>
       </div>
    """

    if st.button("Predict"):
        output = predimage(image)
        st.success(
            "The probability of log's originality is {:.2f} %".format(output))

        if output >= SFACTOR:
            st.markdown(real, unsafe_allow_html=True)
        else:
            st.markdown(fake, unsafe_allow_html=True)


# main function to run the application
if __name__ == '__main__':
    main()
