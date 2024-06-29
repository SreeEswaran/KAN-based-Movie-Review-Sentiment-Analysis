import os
import tarfile
import urllib.request

def download_and_extract_imdb_dataset():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    data_dir = "data/imdb_reviews"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    
    if not os.path.exists(file_path):
        print("Downloading IMDb dataset...")
        urllib.request.urlretrieve(url, file_path)
    
    print("Extracting IMDb dataset...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("Dataset ready.")

if __name__ == "__main__":
    download_and_extract_imdb_dataset()
