import numpy as np

from utils import *
import sklearn
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Flatten, Dense
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from deepface import DeepFace



def show_image(image, landmarks):
    #landmarks = landmarks * 48 + 48  # undo the normalization
    for x in range(0, len(landmarks), 2):
        center = (int(landmarks[x]), int(landmarks[x + 1]))
        image = cv2.circle(image, center, radius=0, color=(0, 0, 255), thickness=-1)

    cv2.imshow("image1", image)
    cv2.waitKey(0)


def scale_rotate_transform(data, labels, rotation_range=10, scale_range=.1):
    '''
    Scales and rotates an image and the keypoints.
    '''
    aug_data = np.copy(data)
    aug_labels = np.copy(labels)

    # Apply rotation and scale transform
    for i in range(len(data)):
        # Unnormalize the keypoints
        aug_labels[i] = aug_labels[i] * 48 + 48
        scale_factor = 1.0 + (np.random.uniform(-1, 1)) * scale_range
        rotation_factor = (np.random.uniform(-1, 1)) * rotation_range

        # Use openCV to get a rotation matrix
        M = cv2.getRotationMatrix2D((48, 48), rotation_factor, scale_factor)
        aug_data[i] = np.expand_dims(cv2.warpAffine(np.squeeze(aug_data[i]), M, (96, 96)), axis=2)
        for j in range(15):
            coord_idx = 2 * j
            old_coord = aug_labels[i][coord_idx:coord_idx + 2]
            new_coord = np.matmul(M, np.append(old_coord, 1))
            aug_labels[i][coord_idx] = new_coord[0]
            aug_labels[i][coord_idx + 1] = new_coord[1]
        # normalize aug_labels
        aug_labels[i] = (aug_labels[i] - 48) / 48

    return aug_data, aug_labels


def horizontal_flip(data, labels):
    '''
    Takes a image set and keypoint labels and flips them horizontally.
    '''

    # Flip the images horizontally
    flipped_data = np.copy(data)[:, :, ::-1, :]
    flipped_labels = np.zeros(labels.shape)
    for i in range(data.shape[0]):
        # Flip the x coordinates of the key points
        flipped_labels[i] += labels[i]
        flipped_labels[i, 0::2] *= -1

        # Flip the indices of the left right keypoints
        flip_indices = [
            (0, 2), (1, 3),
            (4, 8), (5, 9), (6, 10), (7, 11),
            (12, 16), (13, 17), (14, 18), (15, 19),
            (22, 24), (23, 25),
        ]

        for a, b in flip_indices:
            flipped_labels[i, a], flipped_labels[i, b] = flipped_labels[i, b], flipped_labels[i, a]

    return flipped_data, flipped_labels


def data_augmentation(data, labels, rotation_range=10, scale_range=.1, h_flip=True):
    '''
        Takes in a the images and keypoints, applys a random rotation and scaling. Then flips the image
        and keypoints horizontally if specified.

    '''

    aug_data, aug_labels = scale_rotate_transform(data, labels, rotation_range, scale_range)
    if h_flip:
        aug_data, aug_labels = horizontal_flip(aug_data, aug_labels)

    return aug_data, aug_labels


def keypoints_func_ting():
    image = cv2.imread('images/test_image_1.jpg')
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # cv2.imshow("image", image)
    # cv2.imshow("image1", gray)

    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

    # Detect the faces in image
    faces = face_cascade.detectMultiScale(image, 4, 6)

    # Print the number of faces detected in the image
    print('Number of faces detected:', len(faces))

    # Make a copy of the orginal image to draw face detections on
    image_with_detections = np.copy(image)

    # Get the bounding box for each detected face
    for (x, y, w, h) in faces:
        # Add a red bounding box to the detections image
        cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # cv2.imshow("image1", image_with_detections)
    # cv2.waitKey(0)

    # Load training set
    X_train, y_train = load_data()
    print("X_train.shape == {}".format(X_train.shape))
    print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
        y_train.shape, y_train.min(), y_train.max()))

    # Load testing set
    X_test, _ = load_data(test=True)
    print("X_test.shape == {}".format(X_test.shape))

    X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

    X_test, y_test = X_train[:500], y_train[:500]
    X_train, y_train = X_train[500:], y_train[500:]

    data_hflip, labels_hflip = data_augmentation(X_train, y_train, 0.0, 0.0, True)

    X_aug = np.concatenate((X_train, data_hflip), axis=0)
    y_aug = np.concatenate((y_train, labels_hflip), axis=0)

    X_train_transformed, y_train_transformed = data_augmentation(X_aug, y_aug, 15.0, .1, False)
    X_train_transformed2, y_train_transformed2 = data_augmentation(X_aug, y_aug, 15.0, .1, False)

    X_aug = np.concatenate((X_aug, X_train_transformed, X_train_transformed2), axis=0)
    y_aug = np.concatenate((y_aug, y_train_transformed, y_train_transformed2), axis=0)

    X_train_transformed, y_train_transformed = data_augmentation(X_aug, y_aug, 15.0, .1, False)
    X_train_transformed2, y_train_transformed2 = data_augmentation(X_aug, y_aug, 15.0, .1, False)

    X_aug = np.concatenate((X_aug, X_train_transformed, X_train_transformed2), axis=0)
    y_aug = np.concatenate((y_aug, y_train_transformed, y_train_transformed2), axis=0)

    model = Sequential()
    # Conv layer1
    model.add(Conv2D(32, 3, strides=(1, 1), padding='same', activation='elu', input_shape=(96, 96, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(.2))
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

    # Conv layer2
    model.add(Conv2D(64, 3, strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(.2))
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

    # Conv layer3
    model.add(Conv2D(128, 3, strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(.2))
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

    # Conv layer4
    model.add(Conv2D(256, 3, strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

    # Conv layer5
    model.add(Conv2D(256, 3, strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

    # Flatten Layer
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())

    # Fully Connected Layer 2
    model.add(Dense(30, activation='elu'))

    learning_rates = [.001, .0001, .00001]
    batch_sizes = [32, 64, 128]
    full_loss_hist = []
    full_val_loss = []
    for lr in learning_rates:
        ## TODO: Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mean_squared_error')
        for batch_size in batch_sizes:
            print('learning rate: {}, batch size: {}'.format(lr, batch_size))
            ## TODO: Train the model
            hist = model.fit(X_aug, y_aug, batch_size=batch_size, epochs=15, validation_data=(X_test, y_test),
                             verbose=2)
            full_loss_hist.append(hist.history['loss'])
            full_val_loss.append(hist.history['val_loss'])
    ## TODO: Save the model as model.h5
    model.save('my_model.h5')

    # show_image(dst,new_keypoints)
    # cv2.imshow("image1",image)
    # cv2.waitKey(0)

#############################################################################################################


def face_keypoint_detector(image):
    '''
        Takes in an image(BGR) and plots the facial bounding box and keypoints on the image.

        Returns the new image, the face bounding box coordinates and the keypoint coordinates.
    '''
    #load model for keypoints detection
    #model = Sequential()
    model = load_model('my_model_full.h5')

    # Convert image to grayscale
    image_copy = np.copy(image)
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    faces = ()
    for i in range(1, 11):
        for j in np.arange(1.25, i, 0.25):
            faces = face_cascade.detectMultiScale(image, j, i)
            if len(faces) != 0:
                break

        if len(faces) != 0:
            break

    faces_keypoints = []

    # Loop through faces
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 120, 150), 3)
        # Crop Faces
        face = gray[y:y + h, x:x + w]

        # Scale Faces to 96x96
        scaled_face = cv2.resize(face, (96, 96), 0, 0, interpolation=cv2.INTER_AREA)

        # Normalize images to be between 0 and 1
        input_image = scaled_face / 255

        # Format image to be the correct shape for the model
        input_image = np.expand_dims(input_image, axis=0)
        input_image = np.expand_dims(input_image, axis=-1)

        # Use model to predict keypoints on image
        landmarks = model.predict(input_image)[0]

        # Adjust keypoints to coordinates of original image
        landmarks[0::2] = landmarks[0::2] * w / 2 + w / 2 + x
        landmarks[1::2] = landmarks[1::2] * h / 2 + h / 2 + y
        faces_keypoints.append(landmarks)

    return image_copy, faces, faces_keypoints



def extractFeauters(img):

    # scale_percent = 40  # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    #
    # # resize image
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    #extract keypoints
    img2, faces, keypoints = face_keypoint_detector(img)

    x = int(keypoints[0][0] - 40)
    y = int(keypoints[0][1] - 30)
    eye1 = img[y:y + 50, x:x + 80]

    x1 = int(keypoints[0][2] - 40)
    y1 = int(keypoints[0][3] - 30)
    eye2 = img[y1:y1 + 50, x1:x1 + 80]

    x2 = int(keypoints[0][20] - 50)
    y2 = int(keypoints[0][21] - 67)
    nose = img[y2:y2 + 85, x2:x2 + 100]

    x3 = int(keypoints[0][24] - 10)
    y3 = int(keypoints[0][25] - 35)
    mouth = img[y3:y3 + 75, x3:x3 + 135]

    x4 = int(keypoints[0][18] - 10)
    y4 = int(keypoints[0][19] - 30)
    eyebrow1 = img[y4:y4 + 35, x4:x4 + 90]

    x5 = int(keypoints[0][12] - 10)
    y5 = int(keypoints[0][13] - 25)
    eyebrow2 = img[y5:y5 + 35, x5:x5 + 90]

    return eye1, eye2, nose, mouth, eyebrow1, eyebrow2


def create_data_set():
    #img = cv2.imread('images/CFD-MF-308-001-N.jpg')
    directory = 'D:\\faculta\\Licenta\\cfd\\CFD Version 3.0\\Images\\CFD-MR'
    directory2 = 'D:\\Licenta\\dataset\\train'
    arr = os.listdir(directory)
    for i in range(0,len(arr)):
            print(i)
       # new = directory+"\\"+arr[i]
        #arr2 = os.listdir(new)
        #for img in arr2:
            img = arr[i]
            if 'jpg' in img:
                if img[15] == 'N':
                    image = cv2.imread(directory+"\\"+img)
                    eye1, eye2, nose, mouth, eyebrow1, eyebrow2 = extractFeauters(image)
                    ethnicity = ""
                    if img[4] == "A":
                        ethnicity = "\\asian_american"
                    elif img[4] == "B":
                        ethnicity = "\\black"
                    elif img[4] == "I":
                        ethnicity = "\\indian_asian"
                    elif img[4] == "L":
                        ethnicity = "\\latino"
                    elif img[4] == "M":
                        ethnicity = "\\multicultural_american"
                    elif img[4] == "W":
                        ethnicity = "\\white"

                    if img[5] == 'F':
                        directory3 = directory2+"\\female"
                    else:
                        directory3 = directory2 + "\\male"

                    directory4 = directory3 + "\\eye1" + ethnicity+"\\"+str(i)+".jpg"
                    cv2.imwrite(directory4,eye1)

                    directory4 = directory3 + "\\eye2" + ethnicity + "\\" + str(i) + ".jpg"
                    cv2.imwrite(directory4, eye2)

                    directory4 = directory3 + "\\nose" + ethnicity + "\\" + str(i) + ".jpg"
                    cv2.imwrite(directory4, nose)

                    directory4 = directory3 + "\\mouth" + ethnicity + "\\" + str(i) + ".jpg"
                    cv2.imwrite(directory4, mouth)

                    directory4 = directory3 + "\\eyebrow1" + ethnicity + "\\" + str(i) + ".jpg"
                    cv2.imwrite(directory4, eyebrow1)

                    directory4 = directory3 + "\\eyebrow2" + ethnicity + "\\" + str(i) + ".jpg"
                    cv2.imwrite(directory4, eyebrow2)




def model_ochi():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    training_set = train_datagen.flow_from_directory('D:\\Licenta\\dataset\\train\\female\\eye1',
                                                     target_size=(50, 80),
                                                     batch_size=32,
                                                     class_mode='binary')

    # Preprocessing the Test set
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('D:\\Licenta\\dataset\\test\\female\\eye1',
                                                target_size=(50, 80),
                                                batch_size=32,
                                                class_mode='binary')

    cnn = tf.keras.models.Sequential()

    # Step 1 - Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu', strides=2,
                                   input_shape=[50, 80, 3]))
    # cnn.add(tf.keras.layers.Dropout(0.5))
    # Step 2 - Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Adding a second convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.Dropout(0.2))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))

    cnn.add(tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.Dropout(0.7))
    # cnn.add(BatchNormalization())
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))

    cnn.add(tf.keras.layers.Conv2D(filters=128, padding='same', kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.Dropout(0.7))
    # cnn.add(BatchNormalization())
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))

    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    cnn.add(tf.keras.layers.Dense(6, activation=tf.nn.softmax))
    opt = Adam(learning_rate=0.001)
    # compile the model with the optimizer, loss function and metrics for evaluation
    cnn.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    # Part 3 - Training the CNN

    # Compiling the CNN
    # cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])

    # Training the CNN on the Training set and evaluating it on the Test set
    r = cnn.fit(x=training_set, validation_data=test_set, epochs=100, batch_size=128)

    cnn.save('model_female_eye1.h5')



def resize_image(image):

    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    faces = ()
    for i in range(4, 9):
        for j in np.arange(1.25, 8, 0.25):
            faces = face_cascade.detectMultiScale(image, j, i)
            if len(faces) == 1:
                break

        if len(faces) == 1:
            break

    x = int(faces[0][0])
    y = int(faces[0][1])
    face = image[y:y + faces[0][2], x:x + faces[0][3]]
    dim = (384, 384)
    resized = cv2.resize(face, dim, 0, 0, interpolation=cv2.INTER_AREA)

    faces2 = ()
    for extra in range(10, 300):
        for i in range(2, 9):
            for j in np.arange(1.25, 8, 0.25):
                faces2 = face_cascade.detectMultiScale(resized, j, i)
                if len(faces2) == 1:
                    break

            if len(faces2) == 1:
                break

        if len(faces2) == 1:
            break

        face = image[y:y + faces[0][2] + extra, x:x + faces[0][3] + extra]
        dim = (384, 384)
        resized = cv2.resize(face, dim, 0, 0, interpolation=cv2.INTER_AREA)

    #faces2 = face_cascade.detectMultiScale(resized, 1, 6)

    return resized


def clasifica_etnie():
    image = cv2.imread('images/gus.jpg')
    resized = resize_image(image)
    eye1, eye2, nose, mouth, eyebrow1, eyebrow2 = extractFeauters(resized)
    result = DeepFace.analyze(image, actions=['gender'])
    classes = {0: 'asian_american', 1: 'black', 2: 'indian_asian', 3: 'latino', 4: 'multicultural_american', 5: 'white'}

    eye1 = eye1 / 255
    res_eye1 = np.expand_dims(eye1, axis=0)

    eye2 = eye2 / 255
    res_eye2 = np.expand_dims(eye2, axis=0)

    eyebrow1 = eyebrow1 / 255
    res_eyebrow1 = np.expand_dims(eyebrow1, axis=0)

    eyebrow2 = eyebrow2 / 255
    res_eyebrow2 = np.expand_dims(eyebrow2, axis=0)

    mouth = mouth / 255
    res_mouth = np.expand_dims(mouth, axis=0)

    nose = nose / 255
    res_nose = np.expand_dims(nose, axis=0)

    model1 = None
    model2 = None
    model3 = None
    model4 = None
    model5 = None
    model6 = None

    if result['gender'] == 'Woman':
        model1 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_female_mouth.h5')
        model2 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_female_nose.h5')
        model3 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_female_eye1.h5')
        model4 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_female_eye2.h5')
        model5 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_female_eyebrow1.h5')
        model6 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_female_eyebrow2.h5')
    else:
        model1 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_male_mouth.h5')
        model2 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_male_nose.h5')
        model3 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_male_eye1.h5')
        model4 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_male_eye2.h5')
        model5 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_male_eyebrow1.h5')
        model6 = load_model('C:\\Users\\Andreea\\opencv\\Scripts\\model_male_eyebrow2.h5')

        # predict the class

    result1 = model1.predict(res_mouth)
    print('mouth: '+classes[np.argmax(result1[0])])

    result2 = model2.predict(res_nose)
    print('nose: ' + classes[np.argmax(result2[0])])

    result3 = model3.predict(res_eye1)
    print('eye1: ' + classes[np.argmax(result3[0])])

    result4 = model4.predict(res_eye2)
    print('eye2: ' + classes[np.argmax(result4[0])])

    result5 = model5.predict(res_eyebrow1)
    print('eyebrow1: ' + classes[np.argmax(result5[0])])

    result6 = model6.predict(res_eyebrow2)
    print('eyebrow2: ' + classes[np.argmax(result6[0], axis=-1)])

    cv2.imshow("image1", image)
    cv2.waitKey(0)

###################################################################################################


#fata detectata e de 384 384


if __name__ == '__main__':
    # image = cv2.imread('D:\\faculta\\Licenta\\cfd\\CFD Version 3.0\\Images\\CFD\\AF-200\\1.jpg')
    # # result = DeepFace.analyze(image, actions=['gender'])
    # # print("Gender: ", result['gender'])
    #


    print("SALUT")
