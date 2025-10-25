# Install libraries (only needed if Colab)
!pip install tensorflow opencv-python matplotlib scikit-learn

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path
data_dir = "/content/fire_smoke_dataset"  # adjust path if needed
categories = ["fire", "smoke", "normal"]

X, y = [], []

for category in categories:
    folder = os.path.join(data_dir, category)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128,128))
            X.append(img)
            y.append(category)
        except:
            pass

X = np.array(X) / 255.0
y = LabelEncoder().fit_transform(y)
y = to_categorical(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(" Dataset loaded and preprocessed")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')   # Fire, Smoke, Normal
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test)
)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.legend()
plt.show()

loss, acc = model.evaluate(X_test, y_test)
print(f" Test Accuracy: {acc*100:.2f}%")

test_path = "/content/fire_smoke_dataset/fire/img_240.jpg"  # change as needed
img = cv2.imread(test_path)
img = cv2.resize(img, (128,128)) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
labels = ["Fire", "Smoke", "Normal"]

print(" Prediction:", labels[np.argmax(pred)])
plt.imshow(cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB))
plt.title(f"Prediction: {labels[np.argmax(pred)]}")
plt.axis('off')
plt.show()

model.save("fire_smoke_model.h5")
print(" Model saved as fire_smoke_model.h5")
