import os
import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model
from keras.layers import Activation, Dense, Input, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import History
from keras import metrics
batch_size = 32
files_directory = './train'
# img_width, img_height = 224, 224
img_width, img_height = 96,96
files = os.listdir(files_directory)
cat_files = [f for f in files if 'cat' in f ]
dog_files = [f for f in files if 'dog' in f]

# print(cat_files[:3])
# print(dog_files[:3])
df_cat = pd.DataFrame({
    'filename': cat_files,
    'label': 'cat',
})
df_dog = pd.DataFrame({
    'filename': dog_files,
    'label': 'dog',
})
df = pd.concat([df_cat, df_dog])
df = df.sample(frac=1).reset_index(drop=True)

df.head(15)
datagen = ImageDataGenerator(
                             #validation_split=0.01,
                             rescale=1./255., 
#                              width_shift_range=0.2,
#                              height_shift_range=0.2,
                             horizontal_flip=True,
                             rotation_range=40,
                             shear_range=0.2,
                             zoom_range=[0.8, 1.2])
train_gen = datagen.flow_from_dataframe(dataframe=df,
                                        directory=files_directory,
                                        x_col='filename',
                                        y_col='label',
                                        target_size=(img_height, img_width),
                                        batch_size=batch_size,
                                        class_mode='binary',
                                        shuffle=True
#                                         subset='training'
                                       )
# valid_gen = datagen.flow_from_dataframe(dataframe=df,
#                                         directory=files_directory,
#                                         x_col='filename',
#                                         y_col='label',
#                                         target_size=(img_height, img_width),
#                                         batch_size=batch_size,
#                                         class_mode='binary',
#                                         shuffle=True,
#                                         subset='validation')
print(train_gen.class_indices)
dog_label = train_gen.class_indices['dog']
batch_features, batch_labels = next(train_gen)
# batch_features, batch_labels = next(valid_gen)

rows = 4
cols = 4
plt.figure(figsize=(24,24))
for i in range(1, rows*cols + 1):
    plt.subplot(rows,cols,i)
    plt.imshow(batch_features[i-1])
    plt.text(0, 0, 'dog' if batch_labels[i-1] == dog_label else 'cat',  # i don't know why this is suddenly getting flipped
             fontsize=24,
             color='r')
plt.show()
def get_base_model(architecture):
    if architecture == 'resnet' or architecture == 'resnet50':
        base_model = ResNet50(weights='./Petimages/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='average')
    # elif architecture == 'vgg19':
    #     base_model = VGG19(weights='../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='average')
    # else:
    #     base_model = VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='average')
    
    for layer in base_model.layers:
        layer.trainable = True
    
    return base_model
def get_model():
    base_model = get_base_model('resnet50')
    X_input = Input(shape=(img_width, img_height, 3), name='input')
    
    X = base_model(X_input)
    X = Flatten()(X)
    X = Dropout(0.3)(X)
    out = Dense(1, activation='sigmoid')(X)
    return Model(X_input, [out])
model = get_model()

all_history = {
    'loss': [],
    'val_loss': [],
    'acc': [],
    'val_acc': []
}
model.compile(optimizer='sgd',
             loss='binary_crossentropy',
             metrics=['accuracy'])
epochs = 5
history = History()
model.fit_generator(train_gen,
                    epochs=epochs,
                    #validation_data=valid_gen,
                    callbacks=[history])
model.save('model-resnet50-final.h5')
all_history['loss'] += history.history['loss']
# all_history['val_loss'] += history.history['val_loss']
all_history['acc'] += history.history['accuracy']
# all_history['val_acc'] += history.history['val_acc']
plt.figure(figsize=(8,4))
plt.plot(all_history['acc'])
plt.plot(all_history['val_acc'])
plt.title('accuracy')
plt.legend(['train acc', 'valid acc'])
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.show()




