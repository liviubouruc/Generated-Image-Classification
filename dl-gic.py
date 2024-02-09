import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator

INPUT_PATH = '/kaggle/input/gic-unibuc-dl-2023/gic-unibuc-dl-2023'


train_images = []
train_labels = []
train_data = np.loadtxt(INPUT_PATH+'/train.csv', dtype=str)
for data in train_data[1:]:
    file, label = data.split(',')
    pimg = PIL.Image.open(INPUT_PATH+'/train_images/'+file)
    img = copy.deepcopy(np.asarray(pimg))
    train_images.append(img)
    pimg.close()

    train_labels.append(label)

val_images = []
val_labels = []
val_data = np.loadtxt(INPUT_PATH+'/val.csv', dtype=str)
for data in val_data[1:]:
    file, label = data.split(',')
    pimg = PIL.Image.open(INPUT_PATH+'/val_images/'+file)
    img = copy.deepcopy(np.asarray(pimg))
    val_images.append(img)
    pimg.close()

    val_labels.append(label)

train_images = np.array(train_images)
val_images = np.array(val_images)
train_labels = np.array(train_labels).astype('float')
val_labels = np.array(val_labels).astype('float')

train_image_data_generator = ImageDataGenerator(
    zoom_range=0.3,
    fill_mode='nearest')
train_image_data_generator.fit(train_images)

augmented_train_images = []
augmented_train_labels = []

for augmented_images, labels in train_image_data_generator.flow(train_images, train_labels, batch_size=len(train_images), shuffle=False):
    augmented_train_images.append(augmented_images)
    augmented_train_labels.append(labels)
    break

augmented_train_images = np.concatenate(augmented_train_images)
augmented_train_labels = np.concatenate(augmented_train_labels)

def resnet_block(x, filters, strides=2):
    x = keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    y = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)

    y = keras.layers.Conv2D(filters, kernel_size=3, padding='same')(y)
    y = keras.layers.BatchNormalization()(y)
    x = keras.layers.Add()([x, y])
    x = keras.layers.Activation('relu')(x)
    return x

def ResNet18(input_shape=(64, 64, 3), num_classes=100):
    input_tensor = keras.layers.Input(shape=input_shape)
        
    out = keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_tensor)


    out = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2)(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(out)

    out = resnet_block(out, filters=64, strides=1)
    out = resnet_block(out, filters=128)
    out = resnet_block(out, filters=256)
    out = resnet_block(out, filters=512)

    out = keras.layers.AveragePooling2D(pool_size=7)(out)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(num_classes, activation='softmax')(out)

    model = keras.models.Model(inputs=input_tensor, outputs=out)
    return model

learning_rates = [0.01, 0.001, 0.0001]
epochs_list = [30, 50, 75]
resnet_results = []

for lr in learning_rates:
    for epochs in epochs_list:
        resnet_model = ResNet18()
        opt = keras.optimizers.Adam(learning_rate=lr) 
        resnet_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        resnet_model.fit(augmented_train_images, augmented_train_labels, epochs=epochs, batch_size=64, validation_data=(val_images, val_labels))
        resnet_results.append(resnet_model.evaluate(val_images, val_labels))
        
for lr in learning_rates:
    for epochs in epochs_list:
        resnet_model = ResNet18()
        opt = keras.optimizers.SGD(learning_rate=lr) 
        resnet_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        resnet_model.fit(augmented_train_images, augmented_train_labels, epochs=epochs, batch_size=64, validation_data=(val_images, val_labels))
        resnet_results.append(resnet_model.evaluate(val_images, val_labels))
        
print(resnet_results)

def CNN(input_shape=(64, 64, 3), num_classes=100):
    input_tensor = keras.layers.Input(shape=input_shape)
    
    out = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3))(input_tensor)
    out = keras.layers.MaxPooling2D()(out)
    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
    out = keras.layers.MaxPooling2D()(out)
    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
    out = keras.layers.MaxPooling2D()(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Dropout(0.35)(out)

    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(256, activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Dropout(0.2)(out)
    out = keras.layers.Dense(100, activation='softmax')(out)

    model = keras.models.Model(inputs=input_tensor, outputs=out)
    return model


learning_rates = [0.01, 0.001, 0.0001]
epochs_list = [50, 75, 100]
cnn_results = []

for lr in learning_rates:
    for epochs in epochs_list:
        cnn_model = CNN()
        opt = keras.optimizers.Adam(learning_rate=lr) 
        cnn_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cnn_model.fit(augmented_train_images, augmented_train_labels, epochs=epochs, batch_size=64, validation_data=(val_images, val_labels))
        cnn_results.append(cnn_model.evaluate(val_images, val_labels))
        
for lr in learning_rates:
    for epochs in epochs_list:
        cnn_model = CNN()
        opt = keras.optimizers.SGD(learning_rate=lr) 
        cnn_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cnn_model.fit(augmented_train_images, augmented_train_labels, epochs=epochs, batch_size=64, validation_data=(val_images, val_labels))
        cnn_results.append(cnn_model.evaluate(val_images, val_labels))

print(cnn_results)

predicted_val_labels = cnn_model.predict(val_images)
predicted_val_labels_hat = np.array([float(x.argmax()) for x in predicted_val_labels])

cm = confusion_matrix(val_labels, predicted_val_labels_hat)
plt.figure(figsize=(15, 15))
sns.heatmap(cm)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print(f1_score(val_labels, predicted_val_labels_hat, average='micro'))

test_images = []
test_data = np.loadtxt(INPUT_PATH+'/test.csv', dtype=str)
for file in test_data[1:]:
    pimg = PIL.Image.open(INPUT_PATH+'/test_images/'+file)
    img = copy.deepcopy(np.asarray(pimg))
    test_images.append(img)
    pimg.close()

test_images = np.array(test_images)
predicted_test_labels = cnn_model.predict(test_images)
output = zip(test_data[1:], predicted_test_labels)
with open("submission.csv", "w") as g:
    g.write("Image,Class\n")
    for predict in output:
        g.write(str(predict[0]) + ',' + str(predict[1].argmax()) + '\n')
print("done")