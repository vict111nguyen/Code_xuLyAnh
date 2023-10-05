import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.keras.utils import to_categorical
import random,shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model

#định nghĩa hàm generator 
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)
#batch_size: số lượng mẫu sẽ được truyền qua mạng VD có 1500 mẫu training và TS băng 100 thì sẽ lấy 100 mấu 
# ra để train rồi lại lấy tiếp cho đến hết, mục đích giảm dung lượng bộ nhớ yêu cầu

# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 bộ lọc tích chập sử dụng của sổ kích thước 3x3
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#64 convolution filters used each of size 3x3
#chọn các đặc tính tốt nhất thông qua pooling (gộp lại)
   
#bật và tắt ngẫu nhiên các neurons để cải thiện sự hội tụ
    Dropout(0.25),
#Flatten vì quá nhiều kích thước, chúng ta chỉ cần một đầu ra phân loại
    Flatten(),
#fully connected để lấy được đầy đỉ dữ liêu liên quan
    Dense(128, activation='relu'),
#thêm một droppout để tăng sự hôi tụ
    Dropout(0.5),
#xuất ra một softmax để thu gọn ma trận thành các xác suất đầu ra
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)

model.save('models/cnn.h5', overwrite=True)