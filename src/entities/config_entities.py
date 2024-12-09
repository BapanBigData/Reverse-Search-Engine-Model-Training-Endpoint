from from_root import from_root
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()


class DatabaseConfig:
    def __init__(self):
        self.URL: str = os.environ["MONGODB_CONN_STRING"]
        self.DBNAME: str = "reverse_search_engine"
        self.COLLECTION: str = "embeddings"

    def get_database_config(self):
        return self.__dict__


class DataIngestionConfig:
    def __init__(self):
        self.PREFIX: str = "images/"
        self.RAW: str = "data/raw"
        self.SPLIT: str = "data/splitted"
        self.BUCKET: str = os.environ["AWS_BUCKET_NAME"]
        self.SEED: int = 1337
        self.RATIO: tuple = (0.8, 0.1, 0.1)

    def get_data_ingestion_config(self):
        return self.__dict__


class DataPreprocessingConfig:
    def __init__(self):
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 256
        self.TRAIN_DATA_PATH = os.path.join(from_root(), "data", "splitted", "train")
        self.TEST_DATA_PATH = os.path.join(from_root(), "data", "splitted", "test")
        self.VALID_DATA_PATH = os.path.join(from_root(), "data", "splitted", "val")

    def get_data_preprocessing_config(self):
        return self.__dict__


class ModelConfig:
    def __init__(self):
        self.LABEL = 101

    def get_model_config(self):
        return self.__dict__


class TrainerConfig:
    def __init__(self):
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth")
        self.EPOCHS = 5
        self.Evaluation = True

    def get_trainer_config(self):
        return self.__dict__


class ImageFolderConfig:
    def __init__(self):
        self.ROOT_DIR = os.path.join(from_root(), "data", "raw", "images")
        self.IMAGE_SIZE = 256
        self.LABEL_MAP = {}
        self.BUCKET: str = os.environ["AWS_BUCKET_NAME"]
        self.S3_LINK = "https://{0}.s3.ap-south-1.amazonaws.com/images/{1}/{2}"

    def get_image_folder_config(self):
        return self.__dict__


class EmbeddingsConfig:
    def __init__(self):
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth")

    def get_embeddings_config(self):
        return self.__dict__


class AnnoyConfig:
    def __init__(self):
        self.EMBEDDING_STORE_PATH = os.path.join(from_root(), "data", "embeddings", "embeddings.ann")

    def get_annoy_config(self):
        return self.__dict__


class s3Config:
    def __init__(self):
        self.ACCESS_KEY_ID = os.environ["ACCESS_KEY_ID"]
        self.SECRET_KEY = os.environ["AWS_SECRET_KEY"]
        self.REGION_NAME = "us-east-1"
        self.BUCKET_NAME = os.environ["MODEL_REGISTRY"]
        self.KEY = "models"
        self.ZIP_NAME = "artifacts.tar.gz"
        
        self.ZIP_PATHS = [
                (os.path.join(from_root(), "data", "embeddings", "embeddings.json"), "embeddings.json"),
                (os.path.join(from_root(), "data", "embeddings", "embeddings.ann"), "embeddings.ann"),
                (os.path.join(from_root(), "model", "finetuned", "model.pth"), "model.pth")
                        ]

    def get_s3_config(self):
        return self.__dict__