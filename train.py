import numpy as np
from sklearn.tree import DecisionTreeClassifier
# import matplotlib.image as mpimg
# from skimage import io, color, transform, img_as_ubyte, img_as_float
from PIL import Image
from ensemble import AdaBoostClassifier
from feature import NPDFeature
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

imgs = []
img_labels = []
img_features = []
WEAKERS_LIMIT = 5


def load_img():
    for i in range(0, 500):
        with Image.open("./datasets/original/face/face_"+"{:0>3d}".format(i)+".jpg") as image:
            image = image.convert('L')
            image = image.resize((24, 24))
            imgs.append(np.array(image))
            img_labels.append(1)
        with Image.open("./datasets/original/nonface/nonface_" + "{:0>3d}".format(i) + ".jpg") as image:
            image = image.convert('L')
            image = image.resize((24, 24))
            imgs.append(np.array(image))
            img_labels.append(-1)


def npd_feature():
    for i in range(0, len(imgs)):
        print(i)
        features = NPDFeature(imgs[i]).extract()
        img_features.append(features)
 
    
if __name__ == "__main__":
    load_img()
    npd_feature()
    img_features = np.array(img_features)
    img_labels = np.array(img_labels).reshape((-1, 1))
    print(img_features.shape)
    print(img_features)
    X_train, X_val, y_train, y_val = train_test_split(img_features, img_labels, test_size=0.25)
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    ada = AdaBoostClassifier(DecisionTreeClassifier, WEAKERS_LIMIT)
    ada.fit(X_train, y_train)

    y_predict = ada.predict(X_val)
    acc = ada.predict_scores(X_val, y_val)

    print(acc)

    y_val = np.array(list(map(lambda x: int(x), y_val.reshape(1, -1)[0])))
    y_predict = np.array(list(map(lambda x: int(x), y_predict.reshape(1, -1)[0])))

    print(y_predict)
    print(y_val)

    reportContent = 'Accuracy = ' + str(acc) + '\n'
    reportContent += classification_report(y_val, y_predict)

    with open('report.txt', 'w') as report:
        report.write(reportContent)

    pass
        