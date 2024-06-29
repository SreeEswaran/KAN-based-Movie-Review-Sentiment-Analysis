# KAN based Movie Review Sentiment Analysis
This project showcases the implementation of a Knowledge-Aware Neural Network (KAN) for sentiment analysis on movie reviews using the IMDb dataset. The KAN model integrates traditional neural network capabilities with knowledge from a graph to enhance the sentiment classification task.

### Objective:

The goal of this project is to demonstrate how integrating external knowledge through a knowledge graph can enhance the performance of neural networks in natural language processing tasks, specifically sentiment analysis. This project is a valuable resource for researchers and practitioners looking to explore the integration of knowledge graphs with neural networks for improved performance in various NLP tasks.

### Key Features:

1. **Knowledge-Aware Neural Network (KAN):** Incorporates external knowledge to improve sentiment analysis accuracy.
2. **IMDb Movie Reviews Dataset:** Utilizes a widely recognized dataset for sentiment analysis, enabling robust evaluation and benchmarking.
3. **Comprehensive Directory Structure:** Organized codebase with separate modules for data preparation, model definition, training, and evaluation.
4. **Interactive Jupyter Notebook:** Provides an interactive environment for exploring and experimenting with the model.

### Usage:

1. **Data Preparation:** Downloads and preprocesses the IMDb movie reviews dataset.
2. **Model Training:** Trains the KAN model on the preprocessed data.
3. **Evaluation:** Evaluates the trained model on a test dataset and visualizes performance metrics.


## Getting Started

### Prerequisites

- Python 3.7 or later
- TensorFlow 2.x
- NetworkX
- Scikit-learn

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SreeEswaran/KAN-based-Movie-Review-Sentiment-Analysis.git
    cd KAN-based-Movie-Review-Sentiment-Analysis
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the IMDb dataset:
    ```bash
    python data/download_data.py
    ```

### Usage

1. To train the model:
    ```bash
    python src/train_model.py
    ```

2. To evaluate the model:
    ```bash
    python src/evaluate_model.py
    ```

3. To explore the notebook:
    ```bash
    jupyter notebook notebooks/KAN_Movie_Review_Sentiment.ipynb
    ```

### Dataset

The IMDb dataset is a large movie review dataset for binary sentiment classification. You can download it [here](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).

### Model

The model is a Knowledge-Aware Neural Network (KAN) that integrates knowledge from a knowledge graph to enhance the learning process. The architecture includes an LSTM layer for sequence processing and an embedding layer that incorporates knowledge from the graph.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

