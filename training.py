import numpy as np
import os
import cv2
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report
def load_images_from_folder(folder, img_size=(48, 48)):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    class_map = {class_name: idx for idx, class_name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = os.path.join(folder, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
                images.append(img)
                labels.append(class_map[class_name])

    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels, class_map

# Load dataset
train_dir = "/Users/ahmetyigitturhan/PycharmProjects/staj/archive/train"
test_dir = "/Users/ahmetyigitturhan/PycharmProjects/staj/archive/test"
X_train, y_train, class_map = load_images_from_folder(train_dir)
X_test, y_test, _ = load_images_from_folder(test_dir)
# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=len(class_map))
y_test = to_categorical(y_test, num_classes=len(class_map))
# Create the CNN model
def FER_Model(input_shape=(48, 48, 1), num_classes=7):
    visible = Input(shape=input_shape, name='input')
    # 1st block
    conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), name='pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name='drop1_1')(pool1_1)
    # 2nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), name='pool2_1')(conv2_2)
    drop2_1 = Dropout(0.3, name='drop2_1')(pool2_1)
    # 3rd block
    conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_1')(drop2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), name='pool3_1')(conv3_2)
    drop3_1 = Dropout(0.3, name='drop3_1')(pool3_1)
    # 4th block
    conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_1')(drop3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    pool4_1 = MaxPooling2D(pool_size=(2, 2), name='pool4_1')(conv4_2)
    drop4_1 = Dropout(0.3, name='drop4_1')(pool4_1)
    # 5th block
    conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_1')(drop4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    pool5_1 = MaxPooling2D(pool_size=(2, 2), name='pool5_1')(conv5_2)
    drop5_1 = Dropout(0.3, name='drop5_1')(pool5_1)
    # Flatten and output
    flatten = Flatten(name='flatten')(drop5_1)
    output = Dense(num_classes, activation='softmax', name='output')(flatten)
    # Create model
    model = Model(inputs=visible, outputs=output)
    # Summary
    model.summary()
    return model

# Initialize the model
model = FER_Model(input_shape=(48, 48, 1), num_classes=len(class_map))
# Compile the model
opt = Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=40, batch_size=64, verbose=1, validation_split=0.2)
# Evaluate on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)
y_true = np.argmax(y_test, axis=-1)
print(classification_report(y_true, y_pred, target_names=list(class_map.keys())))
model.save('fer_model.h5')