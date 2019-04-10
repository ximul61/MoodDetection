# import time
# from datetime import *
#
# d = datetime.now().strftime('(%d__%H:%M:%S)')
#
#
#
import matplotlib.pyplot as plt, tensorflow as tf
import tensorflow
import tensorflow as tf
#
import cv2
#
# cap = cv2.VideoCapture(0)
#
# if cap.isOpened():
#     _, ff = cap.read()
#     cap.release()  # releasing camera immediately after capturing picture
#     if _ and ff is not None:
#         plt.imshow(ff)
#         plt.show()
#         image2='image_{}.jpg'.format(datetime.strptime(d, '(%d__%H:%M:%S)').strftime("(%I:%M %p)"))
#         cv2.imwrite(image2, ff)

# defination = ['Angrily Disgusted', 'Annoyed', 'Disgust', 'Fearfully Surprised', 'Happy', 'Surprised']
defination=['cats','dogs']
def convert(image):
    img_size = 80
    img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, (img_size, img_size))
    return new_img_array.reshape(-1, img_size, img_size, 1)


model = tensorflow.keras.models.load_model('Final_1d_128ls_4cnn')

prediction = model.predict([convert('19.png')])
print(defination[int(prediction[0][0])])

