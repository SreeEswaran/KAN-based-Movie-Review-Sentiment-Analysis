import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_preparation import load_data

# Load data
_, X_test, _, y_test, _, _ = load_data()

# Load model
model = load_model('../models/kan_movie_review_model.h5')

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy: {accuracy*100:.2f}%")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
