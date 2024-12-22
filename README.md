# Trash Classification Using Deep Learning

## Project Overview

This project demonstrates how to classify images of trash into different categories using deep learning techniques. The model is trained to recognize six types of waste: cardboard, glass, metal, paper, plastic, and trash, based on images. The data is sourced from the `garythung/trashnet` dataset hosted on Hugging Face, and the model is built using a Convolutional Neural Network (CNN) architecture with TensorFlow/Keras.

## Dataset

The dataset used in this project is the **TrashNet** dataset, which contains labeled images of various types of trash. The dataset is divided into six categories:

- **Cardboard**
- **Glass**
- **Metal**
- **Paper**
- **Plastic**
- **Trash**

Each image is labeled with one of these categories, making the task a **multi-class classification** problem.

The dataset is loaded directly from Hugging Face using the `datasets` library and is split into training and testing sets. The dataset is further augmented for better generalization.

### Data Preprocessing

1. **Saving Images Locally**: Images are downloaded and saved locally using a custom function. This is essential for use with `ImageDataGenerator`, which requires file paths to work.
2. **Data Augmentation**: Data augmentation is applied to the training data to improve the model's generalization ability. The augmentations include random rotations, width/height shifts, flips, zooming, and brightness adjustments.
3. **Data Splitting**: The dataset is split into 80% training and 20% testing using `train_test_split` from `datasets`.

## Installation

To run the project, you need to install the following dependencies:

```bash
pip install datasets tensorflow matplotlib numpy seaborn scikit-learn pandas
```

Additionally, if you're working in a Jupyter environment, use the following to ensure correct visualization:

```bash
pip install jupyter
```

## Project Structure

```bash
/
├── model_architecture_CNN.png    # Image showing the CNN model architecture
├── trash_classifier_model.h5     # Trained deep learning model (saved after training)
└── README.md                     # This file
```

## Model Architecture

The model is built using a Convolutional Neural Network (CNN), which is a type of deep learning model commonly used for image classification tasks. The architecture is as follows:

1. **Conv2D Layers**: Convolutional layers extract features from the images by applying filters. These filters help detect various patterns such as edges, textures, and shapes in the image. The filters are learned during the training process, allowing the model to automatically recognize relevant features.
2. **MaxPooling2D Layers**: These layers downsample the spatial dimensions, reducing computational complexity while retaining essential features. Max pooling helps make the model more invariant to small translations and distortions in the image.
3. **Flatten Layer**: This layer flattens the 2D output of the convolutional layers into a 1D array for classification. It prepares the features for input to the fully connected layers.
4. **Dense Layers**: These fully connected layers interpret the extracted features and make predictions based on the learned features. The more units in the dense layer, the more complex the model is.
5. **Output Layer**: The output layer has 6 units (one for each class), with a softmax activation function, which outputs probabilities for each class. The softmax function ensures that the outputs are normalized into a probability distribution.

The model is compiled with the Adam optimizer and categorical crossentropy loss, as this is a multi-class classification problem.

### CNN Model:

```python
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])
    return model
```

### Compilation:

```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

### Model Summary:

```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 222, 222, 32)      896
max_pooling2d (MaxPooling2D) (None, 111, 111, 32)      0
conv2d_1 (Conv2D)            (None, 109, 109, 64)      18496
max_pooling2d_1 (MaxPooling (None, 54, 54, 64)        0
conv2d_2 (Conv2D)            (None, 52, 52, 128)      73856
max_pooling2d_2 (MaxPooling (None, 26, 26, 128)      0
flatten (Flatten)            (None, 85888)             0
dense (Dense)                (None, 128)               10913952
dropout (Dropout)            (None, 128)               0
dense_1 (Dense)              (None, 6)                 774
=================================================================
```

## Model Training

The model is trained for **50 epochs** using a batch size of 32. The training and validation accuracy, as well as the loss values, are plotted during training to monitor the model's performance.

```python
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=test_generator.samples // test_generator.batch_size
)
```

## Evaluation and Performance

After training, the model is evaluated on the testing set:

```python
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```

### Confusion Matrix & Classification Report

The model's performance is further evaluated using a **confusion matrix** and **classification report**, which provide insights into the model's accuracy, precision, recall, and F1-score for each class.

```python
print("
Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
```

## Visualization

Random test samples are visualized with their true and predicted labels to evaluate how well the model performs on unseen data.

```python
plt.figure(figsize=(16, 9))
random_indices = np.random.choice(len(test_data["images"]), size=16, replace=False)
```

## Saving the Model

After training, the model is saved in the **H5** format for later use:

```python
model.save("trash_classifier_model.h5")
```

## Conclusion

This project provides a robust solution for classifying trash into different categories using deep learning. The use of CNNs allows for automatic feature extraction from the images, and data augmentation improves the model's ability to generalize. The evaluation metrics ensure that the model performs well across all six classes.
