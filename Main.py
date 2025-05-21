import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_pokedex(description_file, image_folder):
    pokedex = pd.read_csv(description_file)
    if 'Type2' in pokedex.columns:
        pokedex.drop('Type2', axis=1, inplace=True)
    pokedex.sort_values(by=['Name'], ascending=True, inplace=True)
    images = sorted(os.listdir(image_folder))
    images = [os.path.join(image_folder, img) for img in images]
    pokedex['Image'] = images
    return pokedex

def prepare_dataset(pokedex):
    data_gen = ImageDataGenerator(
        validation_split=0.1,
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_gen = data_gen.flow_from_dataframe(
        dataframe=pokedex,
        x_col='Image',
        y_col='Type1',
        target_size=(120, 120),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        color_mode='rgba'
    )

    val_gen = data_gen.flow_from_dataframe(
        dataframe=pokedex,
        x_col='Image',
        y_col='Type1',
        target_size=(120, 120),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='rgba'
    )

    return train_gen, val_gen

def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(120,120,4)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def main(description_file, image_folder, epochs=30):
    pokedex = load_pokedex(description_file, image_folder)
    print(pokedex.info())

    train_gen, val_gen = prepare_dataset(pokedex)

    num_classes = len(train_gen.class_indices)
    print(f"Liczba klas: {num_classes}")

    model = build_model(num_classes)
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    test_loss, test_acc = model.evaluate(val_gen)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Dokładność')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Strata')
    plt.legend()

    plt.show()

    epochs_ran = len(history.history['accuracy'])
    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy'])
    mean_train_acc = np.mean(history.history['accuracy'])
    median_train_acc = np.median(history.history['accuracy'])

    print("\n=== Podsumowanie treningu ===")
    print(f"Liczba epok: {epochs_ran}")
    print(f"Najlepsza dokładność treningowa: {best_train_acc:.4f}")
    print(f"Najlepsza dokładność walidacyjna: {best_val_acc:.4f}")
    print(f"Średnia dokładność treningowa: {mean_train_acc:.4f}")
    print(f"Mediana dokładności treningowej: {median_train_acc:.4f}")
    print(f"\nDokładność na zbiorze walidacyjnym: {test_acc:.4f}")
    print(f"Strata na zbiorze walidacyjnym: {test_loss:.4f}")

if __name__ == '__main__':
    description_file = 'pokemon_dataset/pokemon.csv'
    image_folder = 'pokemon_dataset/images/images'
    main(description_file, image_folder)
