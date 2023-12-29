"""Weather Recognition Project"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, VGG19, Xception, ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetLarge, NASNetMobile
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Set the path to the directory containing the training images
path = "./data"
path_imgs = list(glob.glob(path+"/**/*.jpg"))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], path_imgs))
# print(labels) # labels of each file that was read
file_path = pd.Series(path_imgs, name="File_Path").astype(str)
labels = pd.Series(labels, name="Labels")
data = pd.concat([file_path, labels], axis=1)
data = data.sample(frac=1).reset_index(drop=True) # shuffle the data
print(data.head())

# Visualize the data
def show_initial_data(data):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8),
                         subplot_kw={'xticks': [], 'yticks': []})   
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(data.File_Path[i]))
        ax.set_title(data.Labels[i])
    plt.tight_layout()
    plt.show()

show_initial_data(data)

# Visualize the counts
def show_counts(data):
    fig, ax = plt.subplots()
    data.Labels.value_counts().plot(kind='bar', ax=ax)
    plt.show()

show_counts(data)

# Split the data into Train and Test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)  

# Functions 
# Generators
def gen(preprocessor, train, test):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocessor,
                                       validation_split=0.2)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessor)

    train_gen = train_datagen.flow_from_dataframe(dataframe=train,
                                                    x_col="File_Path",
                                                    y_col="Labels",
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    subset="training",
                                                    shuffle=True,
                                                    rotation_range=30,
                                                    shear_range=0.2,
                                                    fill_mode="nearest",
                                                    horizontal_flip=True,
                                                    seed=42)
    valid_gen = train_datagen.flow_from_dataframe(dataframe=train,
                                                    x_col="File_Path",
                                                    y_col="Labels",
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    subset="validation",
                                                    shuffle=False,
                                                    rotation_range=30,
                                                    shear_range=0.2,
                                                    fill_mode="nearest",
                                                    horizontal_flip=True,
                                                    seed=42)
    test_gen = test_datagen.flow_from_dataframe(dataframe=test,
                                                x_col="File_Path",
                                                y_col="Labels",
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode="categorical",
                                                shuffle=False,
                                                seed=42)
    return train_gen, valid_gen, test_gen

def func(name_model):
    pre_model = name_model(input_shape = (224, 224, 3), 
                           include_top = False, 
                           weights = 'imagenet',
                           pooling='avg')
    pre_model.trainable = False
    inputs = pre_model.input
    x = Dense(128, activation='relu')(pre_model.output)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(11, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    my_callbacks = [EarlyStopping(monitor='val_loss', 
                                  min_delta=0, 
                                  patience=5, 
                                  mode='auto')]
    return model, my_callbacks

def plot(history, test_gen, train_gen, model):
    # Plotting accuracy, validation_accuracy, loss, validation_loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()    

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])

    # Predict on Test Data
    pred = model.predict(test_gen)
    pred = np.argmax(pred, axis=1)
    labels = (train_gen.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]

    # Classification report
    clr = classification_report(test_df.Labels, pred)   
    print("Classification Report:\n----------------------\n", clr)

    # Confusion matrix
    cm = confusion_matrix(test_df.Labels, pred)
    print("Confusion Matrix:\n----------------------\n", cm)
    ConfusionMatrixDisplay.from_predictions(test_df.Labels, pred, 
                                            colorbar=False
                                            )

    # Display 6 pictures of the dataset with their labels
    fig, axes = plt.subplots(4, 3, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(test_df.File_Path.iloc[i+1]))
        ax.set_title(f"True: {test_df.Labels.iloc[i+1]}\nPredicted: {pred[i+1]}")
    plt.tight_layout()
    plt.show()

    return history

def result_test(test, model_use):
    results = model_use.evaluate(test, verbose=0)
    print("    Test Loss: {:.5f}".format(results[0]))
    print("Test Accuracy: {:.2f}%".format(results[1]*100))

    return results

###############################
# Train Models
###############################

# EfficientNetB7
# from tensorflow.keras.applications import EfficientNetB7
# from tensorflow.keras.applications.efficientnet import preprocess_input
# ENet_preprocessor = preprocess_input
# train_gen_Enet, valid_gen_Enet, test_gen_Enet = gen(ENet_preprocessor, train_df, test_df)   
# ENet_model, callback = func(EfficientNetB7)
# history = ENet_model.fit(train_gen_Enet, 
#                               validation_data=valid_gen_Enet, 
#                               epochs=10, 
#                               callbacks=callback)

# history_Enet = plot(history, test_gen_Enet, train_gen_Enet, ENet_model)

# result_Enet = result_test(test_gen_Enet, ENet_model)

# Resnet50
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# ResNet_pre=preprocess_input
# train_gen_ResNet, valid_gen_ResNet, test_gen_ResNet = gen(ResNet_pre,train_df,test_df)
# ResNet_model, callback=func(ResNet50)
# history = ResNet_model.fit(
#     train_gen_ResNet,
#     validation_data=valid_gen_ResNet,
#     epochs=10,
#     callbacks=callback,
# )
# history_ResNet= plot(history,test_gen_ResNet,train_gen_ResNet, ResNet_model)
# result_ResNet = result_test(test_gen_ResNet,ResNet_model)
# print("Completed training ResNet50")

# MobileNet
# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.applications.mobilenet import preprocess_input
# MobileNet_pre=preprocess_input
# train_gen_MobileNet, valid_gen_MobileNet, test_gen_MobileNet = gen(MobileNet_pre,train_df,test_df)
# MobileNet_model, callback=func(MobileNet)
# history = MobileNet_model.fit(
#     train_gen_MobileNet,
#     validation_data=valid_gen_MobileNet,
#     epochs=10,
#     callbacks=callback,
# )
# history_MobileNet = plot(history,test_gen_MobileNet,train_gen_MobileNet, MobileNet_model)
# result_MobileNet = result_test(test_gen_MobileNet,MobileNet_model)
# print("Completed training MobileNet")

# MobileNetv2
# from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
MobileNetv2_pre=preprocess_input
train_gen_MobileNetv2, valid_gen_MobileNetv2, test_gen_MobileNetv2 = gen(MobileNetv2_pre,train_df,test_df)
MobileNetv2_model, callback=func(MobileNet)
history = MobileNetv2_model.fit(
                        train_gen_MobileNetv2,
                        validation_data=valid_gen_MobileNetv2,
                        epochs=10,
                        callbacks=callback,
                    )
history_MobileNet = plot(history,test_gen_MobileNetv2,train_gen_MobileNetv2, MobileNetv2_model)
result_MobileNet = result_test(test_gen_MobileNetv2,MobileNetv2_model)
print("Completed training MobileNetv2")



# VGG19
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.applications.vgg19 import preprocess_input
# VGG19_pre=preprocess_input
# train_gen_VGG19, valid_gen_VGG19, test_gen_VGG19 = gen(VGG19_pre,train_df,test_df)
# VGG19_model, callback=func(VGG19)
# history = VGG19_model.fit(
#     train_gen_VGG19,
#     validation_data=valid_gen_VGG19,
#     epochs=10,
#     callbacks=callback,
# )
# history_VGG19= plot(history,test_gen_VGG19,train_gen_VGG19, VGG19_model)
# result_VGG19 = result_test(test_gen_VGG19,VGG19_model)
# print("Completed training VGG19")

# Xception
# from tensorflow.keras.applications import Xception
# from tensorflow.keras.applications.xception import preprocess_input
# Xception_pre=preprocess_input
# train_gen_Xception, valid_gen_Xception, test_gen_Xception = gen(Xception_pre,train_df,test_df)
# Xception_model, callback=func(Xception)
# history = Xception_model.fit(
#     train_gen_Xception,
#     validation_data=valid_gen_Xception,
#     epochs=10,
#     callbacks=callback,
# )
# history_Xception = plot(history,test_gen_Xception,train_gen_Xception, Xception_model)
# result_Xception = result_tresult_Xception = result_test(test_gen_Xception,Xception_model)
# print("Completed training Xception")


# Final Report of all models
output = pd.DataFrame({'Model': ['EfficientNetB7',
                                 'Resnet50', 
                                 'MobileNet',
                                 'VGG19', 
                                 'Xception'],
                        'Accuracy': [result_Enet[1], 
                                     result_ResNet[1], 
                                     result_MobileNet[1],
                                     result_VGG19[1], 
                                     result_Xception[1]]})

# plot result
fig, ax = plt.subplots()
output.plot(kind='bar', ax=ax)
plt.show()

# Save the model
MobileNetv2_model.save('./MobileNetv2.h5')
MobileNetv2_model.save('MobileNetv2.keras')
print("Saved model to disk")