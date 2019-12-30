#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)


# In[2]:


cwd = r"D:\Narendra\AIGenie_Capstone_ALL\soccerball\\"


# In[50]:


def load_images_from_folder(folder):
    filenames = []
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            images.append(image)
            filenames.append(filename)
    return {"images":images, "filenames":filenames}


# In[51]:


pos_data = load_images_from_folder(cwd + "soccerball")
print(len(pos_data['images']), pos_data['images'][0].shape)


# In[52]:


neg_data = load_images_from_folder(cwd + "neg_images")
print(len(neg_data['images']), neg_data['images'][0].shape)


# In[53]:


plt.imshow(pos_data['images'][3])

plt.show()
# In[54]:


plt.imshow(neg_data['images'][3])
plt.show()

# In[55]:


pos_data['label'] = np.ones(len(pos_data['images']))
neg_data['label'] = np.zeros(len(neg_data['images']))


# In[56]:


data = {}
pos_data['images'].extend(neg_data['images'])
data['images'] = np.array(pos_data['images'])


# In[57]:


data_shape = data['images'].shape

# In[58]:


data['label'] = np.concatenate((pos_data['label'], neg_data['label']), axis = 0).reshape(data_shape[0], 1)
print(data['label'].shape)


# In[59]:


print(data_shape)


# In[60]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data['images'], data['label'], shuffle = True, stratify = data['label'])


# In[61]:


ntrain = len(train_x)
nval = len(test_x)
print(ntrain, nval)


# In[62]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 40, 
                                  width_shift_range = 0.2, height_shift_range = 0.2, 
                                  shear_range = 0.2, zoom_range = 0.2, 
                                  horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)


# In[63]:


train_generator = train_datagen.flow(train_x, train_y, batch_size = 16)
val_generator = val_datagen.flow(test_x, test_y, batch_size = 16)


# In[67]:


from keras import applications
from keras.models import Sequential,Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras import layers
from keras.optimizers import Adam

"""
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (64,64,3)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "softmax"))
opt = Adam(lr = 0.00001)
model.compile(optimizer = opt,
             loss = 'binary_crossentropy',
             metrics = ['acc'])


# In[65]:


model.summary()


# In[ ]:


model.fit_generator(train_generator, epochs = 20, steps_per_epoch = ntrain//16, 
                    validation_data = val_generator, validation_steps = nval//16)


# In[47]:
"""

base_model = applications.resnet.ResNet50(include_top=False, weights='imagenet', 
                                                  input_shape=(128, 128, 3), pooling=None, classes=1)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation= 'relu')(x)
x = Dense(128, activation= 'relu')(x)
predictions = Dense(1, activation= 'softmax')(x)
resnet_model = Model(inputs = base_model.input, outputs = predictions)


# In[48]:


resnet_model.compile(optimizer = 'adam',
             loss = 'binary_crossentropy',
             metrics = ['acc'])


# In[49]:


training_history = resnet_model.fit_generator(train_generator, epochs = 5, steps_per_epoch = ntrain//16, 
                    validation_data = val_generator, validation_steps = nval//16)


# In[ ]:




