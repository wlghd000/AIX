# AIX

주제 : 고기굽기정도(레어, 미디움, 웰던) 판별하기

팀 : 이스타니슬라브, 이상민, 이지홍

동기 및 목적 : 요리를 이제 막 시작해보는 사람들에게 스테이크 굽기는 재밌으면서도 어렵다. 미디움으로 충분히 익혔다고 생각한 고기를 잘라보니 레어인지 미디움인지 아리송할때가 많다. 이러한 요리 초보들에게 단면의 사진을 찍기만 하면 굽기가 어느정도인지 판별해주는 장치를 만들고 싶다. 이 프로젝트에서는 스테이크의 단면 이미지를 학습하여 굽기정도를 판별해볼 것이다.

# dataset : 

레어 고기 단면 사진 모음 300장

미디움 고기 단면 사진 모음 30장 

웰던 고기 단면 사진 모음 60장

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

import os

#원본 데이터 경로 및 증강된 데이터 저장 경로 설정

original_dataset_dir = 'c:\\Users\\wlghd\\OneDrive\\바탕 화면\\ai\\train\\medium'

augmented_dir = 'c:\\Users\\wlghd\\OneDrive\\바탕 화면\\ai\\train\\medium'

os.makedirs(augmented_dir, exist_ok=True)

#이미지 증강 및 저장

datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,
 
    width_shift_range=0.2,
  
    height_shift_range=0.2,
 
    shear_range=0.2,
  
    zoom_range=0.2,
  
    horizontal_flip=True

)

#이미지 복사 및 증강하여 저장

def copy_and_augment_images(source, dest):
    
    for file_name in os.listdir(source):
     
        img_path = os.path.join(source, file_name)
       
        img = load_img(img_path, target_size=(150, 150))
       
        x = img_to_array(img)
       
        x = x.reshape((1,) + x.shape)
       
        i = 0
       
        for batch in datagen.flow(x, batch_size=1, save_to_dir=dest, save_prefix='augmented', save_format='jpeg'):
          
            i += 1
           
            if i > 5:  # 증강할 이미지 수 조절 가능
            
                break

copy_and_augment_images(original_dataset_dir, augmented_dir)

위 코드를 실행하여 부족한 사진들을 증가시킴

# cnn 모델 학습 :
keras와 tensorflow 사용

우선 바탕화면에 빈 폴더를 만들고 안에 train과 validation 폴더를 만든다.

후에 각 폴더 안에 rare/medium/welldone 폴더를 만든 후 위에서 얻은 데이터를 저장한다.

vscode에 아래 코드를 입력한다.

  from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

  from tensorflow.keras.models import Sequential

  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

  import numpy as np

  #데이터셋 경로 설정

  train_dir = 'C:\\Users\\wlghd\\OneDrive\\바탕 화면\\ai\\train'

  validation_dir = 'C:\\Users\\wlghd\\OneDrive\\바탕 화면\\ai\\validation'

  #이미지 데이터 전처리 설정

  datagen = ImageDataGenerator(rescale=1./255)

  #데이터셋 불러오기 및 전처리

  train_generator = datagen.flow_from_directory(
   
      train_dir,
    
      target_size=(150, 150),
   
      batch_size=32,
   
      class_mode='categorical'

  )

  validation_generator = datagen.flow_from_directory(
    
      validation_dir,
   
      target_size=(150, 150),
    
      batch_size=32,
   
      class_mode='categorical'

  )

  #CNN 모델 구성

  model = Sequential([
 
      Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
   
      MaxPooling2D((2, 2)),
    
      Conv2D(64, (3, 3), activation='relu'),
   
      MaxPooling2D((2, 2)),
   
      Conv2D(128, (3, 3), activation='relu'),
   
      MaxPooling2D((2, 2)),
   
      Flatten(),
   
      Dense(128, activation='relu'),
   
      Dense(3, activation='softmax')  

  ])

  #모델 컴파일

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  #모델 학습

  model.fit(train_generator, epochs=10, validation_data=validation_generator)

# 이미지 판단 :

새롭게 판단할 이미지를 구한 후 

#새로운 이미지를 모델에 적용하여 예측

def predict_image_class(image_path):
   
    img = load_img(image_path, target_size=(150, 150))
   
    img = img_to_array(img) / 255.0
    
    img = np.expand_dims(img, axis=0)
   
    prediction = model.predict(img)
   
    classes = train_generator.class_indices
   
    predicted_class = list(classes.keys())[np.argmax(prediction)]
   
    print(f"The image is predicted as: {predicted_class}")

#이미지 판단 함수 호출

image_path_to_predict = 'C:\\Users\\wlghd\\Downloads\\finaltest\\test.jpg'  # 판단하고자 하는 이미지의 경로

predict_image_class(image_path_to_predict)

위 코드를 추가로 vs코드에 입력한다. 

# 실행 :

anaconda에서 

pip install tensorflow 를 통해 다운받고

python deeplearning.py 실행한다

# 실행 결과 :

![image](https://github.com/wlghd000/AIX/assets/150144544/e3da6584-d7f0-46d1-be7d-a543be5ae793)

레어 굽기의 사진을 판단하였더니 레어로 판단한 것을 볼 수 있었다.

# 관련 조사
chatgpt의 도움을 받아 코드를 작성하였다.

# 결론
데이터셋 증가를 하기 전후의 정확도가 많이 차이나는 것을 알 수 있었는데, 적절한 데이터셋의 양이 cnn 모델 학습에 중요한 요소임을 깨달았다.
