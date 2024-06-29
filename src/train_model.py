from data_preparation import load_data
from model_definition import build_kan_model, knowledge_graph

# Load data
X_train, X_test, y_train, y_test, word_index, max_length = load_data()

# Build model
model = build_kan_model(max_length, word_index, knowledge_graph)

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('../models/kan_movie_review_model.h5')

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy: {accuracy*100:.2f}%")
