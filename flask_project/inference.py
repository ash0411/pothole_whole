from  tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
model = load_model('my_model3conlayer.h5')
def get_prediction(image_path):
    #image = cv2.resize(image_path,(64,64))
    print("this is image path",image_path)
    image = tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image,axis =0)
    print(image.shape)
    prediction = model.predict(image)
    print(prediction)
    if prediction[0][1] > prediction[0][0]:
        print("pothole")
    else:
        print("not pothole")
    return prediction



