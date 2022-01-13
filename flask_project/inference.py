from  tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
model = load_model('my_model_weights_resnet50_trainable_tt.h5')
def get_prediction(image_path):
    #image = cv2.resize(image_path,(64,64))
    print("this is image path",image_path)
    image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224,224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    #image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image,axis =0)
    print(image.shape)
    prediction = model.predict(image)
    prediction = np.argmax(prediction,axis=1)
    print(prediction)
    # if prediction[0 > prediction[0][0]:
    #     print("pothole")
    # else:
    #     print("not pothole")
    return prediction



