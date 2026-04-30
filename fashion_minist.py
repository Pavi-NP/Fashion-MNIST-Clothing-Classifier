# fashion_minist.py
# ============================================
# CNN Image Classifier for Fashion Categorization
# StyleVision Prototype Model
# Author: Pavi (Computer Vision Engineer Intern)
# ============================================

# pip install numpy -q
# pip install panda -q
# pip install matplotlib -q
# pip install tensorflow -q

# pip install opendatasets -q # Kaggle dataset downloader
# pip install scikit-learn
# pip install seaborn


# fashion_minist.py
# Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================================
# Load Dataset
# ============================================

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ============================================
# Normalize Data
# ============================================

train_images = train_images / 255.0
test_images = test_images / 255.0

# Add channel dimension
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))


# ============================================
# Data Augmentation
# ============================================


data_augmentation = models.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])


# # Better augmentation pipeline
# data_augmentation = models.Sequential([
#     layers.RandomFlip("horizontal"),  # CRITICAL for clothing symmetry
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#     layers.RandomTranslation(0.1, 0.1),
#     layers.RandomBrightness(0.1),  # Added
# ])
# Test Accuracy: 0.6913999915122986

# Deeper architecture with residual connections (simplified example)
# Or switch to ResNet18/50 pre-trained architecture for better performance


# ============================================
# CNN Model Architecture
# ============================================

model = models.Sequential([

    data_augmentation,

    # First convolution block with batch normalization
    layers.Conv2D(32, (3,3), padding='same', activation='relu',
                  input_shape=(28,28,1)),
    layers.BatchNormalization(), # Added batch normalization for better training stability
    layers.MaxPooling2D((2,2)),

    # Second convolution block
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(), # Added batch normalization for better training stability
    layers.MaxPooling2D((2,2)),

    # Third convolution block
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(), # Added batch normalization for better training stability
    #layers.MaxPooling2D((2,2)),
    
    #------
    # # Flatten layer
    # layers.Flatten(),

    # # Dense layers
    # layers.Dense(128, activation='relu'),

    # - - - - - -
    # This is more robust to spatial shifts from augmentation

    layers.GlobalAveragePooling2D(),  # Averages each feature map
    layers.Dense(128, activation='relu'),
    
    #----------

    # Dropout regularization
    layers.Dropout(0.5),

    # Output layer
    layers.Dense(10, activation='softmax')
])


# # Model summary
# model.summary()


# ============================================
# Compile Model
# ============================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================
# callbacks: EarlyStopping & ReduceLROnPlateau
# ============================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, verbose=1
)


# ============================================
# Train Model
# ============================================

history = model.fit(
    train_images,
    train_labels,
    epochs=50, # Increased epochs for better training (20-->50)
    validation_split=0.2,
    callbacks=[early_stop],
    batch_size=16 # Reduced batch size for better convergence (32-->16)
)

# Model summary
model.summary()


# ============================================
# Plot Training Performance
# ============================================

plt.figure(figsize=(10,4))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train","Validation"])

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train","Validation"])

plt.show()


# ============================================
# Evaluate Model
# ============================================

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels
)

print("\nTest Accuracy:", test_accuracy)


# ============================================
# Predictions
# ============================================

predictions = model.predict(test_images)

predicted_labels = np.argmax(predictions, axis=1)


# ============================================
# Confusion Matrix
# ============================================

cm = confusion_matrix(test_labels, predicted_labels)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()


# ============================================
# Example Prediction Visualization
# ============================================

plt.figure(figsize=(10,6))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.imshow(test_images[i].reshape(28,28), cmap='gray')

    predicted_class = class_names[predicted_labels[i]]

    true_class = class_names[test_labels[i]]

    plt.title(f"P:{predicted_class}\nT:{true_class}")

    plt.axis("off")

plt.show()
