import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(x_train, y_train, batch_size=64)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=64)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=30
    )

    test_loss, test_acc = model.evaluate(test_generator)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

    epochs = len(history.history['accuracy'])
    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy'])
    mean_train_acc = np.mean(history.history['accuracy'])
    median_train_acc = np.median(history.history['accuracy'])

    print("\n=== Podsumowanie treningu ===")
    print(f"Liczba epok: {epochs}")
    print(f"Najlepsza dokładność treningowa: {best_train_acc:.4f}")
    print(f"Najlepsza dokładność walidacyjna: {best_val_acc:.4f}")
    print(f"Średnia dokładność treningowa: {mean_train_acc:.4f}")
    print(f"Mediana dokładności treningowej: {median_train_acc:.4f}")
    print(f"\nDokładność na zbiorze testowym: {test_acc:.4f}")
    print(f"Strata na zbiorze testowym: {test_loss:.4f}")

if __name__ == "__main__":
    main()
