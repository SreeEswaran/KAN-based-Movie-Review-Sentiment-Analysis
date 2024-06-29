import numpy as np
import networkx as nx
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

# Create knowledge graph
def create_knowledge_graph():
    knowledge_graph = nx.Graph()
    knowledge_graph.add_edge("good", "positive")
    knowledge_graph.add_edge("excellent", "positive")
    knowledge_graph.add_edge("bad", "negative")
    knowledge_graph.add_edge("terrible", "negative")
    knowledge_graph.add_edge("great", "positive")
    knowledge_graph.add_edge("poor", "negative")
    knowledge_graph.add_edge("fantastic", "positive")
    knowledge_graph.add_edge("horrible", "negative")
    return knowledge_graph

knowledge_graph = create_knowledge_graph()

# Get related nodes
def get_related_nodes(graph, word):
    if word in graph:
        return list(graph.neighbors(word))
    return []

# Get knowledge embedding
def get_knowledge_embedding(word_index, graph, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        related_nodes = get_related_nodes(graph, word)
        if related_nodes:
            related_embeddings = [np.random.rand(embedding_dim) for _ in related_nodes]
            embedding_matrix[i] = np.mean(related_embeddings, axis=0)
        else:
            embedding_matrix[i] = np.random.rand(embedding_dim)
    return embedding_matrix

# Build KAN model
def build_kan_model(max_length, word_index, knowledge_graph, embedding_dim=50):
    text_input = Input(shape=(max_length,), name='text_input')
    embedding_layer = Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, input_length=max_length)(text_input)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)
    lstm_layer = Bidirectional(LSTM(64))(lstm_layer)
    lstm_layer = Dropout(0.5)(lstm_layer)

    knowledge_embedding_matrix = get_knowledge_embedding(word_index, knowledge_graph, embedding_dim)
    knowledge_embedding_layer = Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, weights=[knowledge_embedding_matrix], input_length=max_length, trainable=False)(text_input)

    concatenated = Concatenate()([lstm_layer, knowledge_embedding_layer])
    dense_layer = Dense(64, activation='relu')(concatenated)
    dense_layer = Dropout(0.5)(dense_layer)
    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=text_input, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
