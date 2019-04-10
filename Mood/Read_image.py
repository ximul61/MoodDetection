import pickle
import random

import numpy as np, os, cv2

imdir = '/home/shimul/Install_Soft/Emotions'

categories = ['Angrily Disgusted', 'Annoyed', 'Disgust', 'Fearfully Surprised', 'Happy', 'Surprised']
img_size = 80

training_image = []


def create_training_image():
    for category in categories:
        path = os.path.join(imdir, category)
        image_set_no = categories.index(category)
        for image in os.listdir(path):
            try:
                image_set = cv2.imread(os.path.join(path, image),cv2.IMREAD_GRAYSCALE)
                new_image_set = cv2.resize(image_set, (img_size, img_size))
                training_image.append([new_image_set, image_set_no])

                # mtb.imshow(new_image_set,cmap='gray')
                # mtb.show()
                # break
                #

            except Exception as E:
                pass
        # break


create_training_image()

print(len(training_image))

random.shuffle(training_image)

for sample in training_image[:10]:
    print(sample[1])

x_feature = []
y_label = []

for feature, label in training_image:
    x_feature.append(feature)
    y_label.append(label)
x_feature = np.array(x_feature).reshape(-1, img_size, img_size, 1)

pickle_out = open('feature.pickle', 'wb')
pickle.dump(x_feature, pickle_out)
pickle_out.close()

pickle_out = open('label.pickle', 'wb')
pickle.dump(y_label, pickle_out)
pickle_out.close()

pickle_in = open('feature.pickle', 'rb')
x_feature = pickle.load(pickle_in)
print(x_feature[1])
