from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.embeddings import EmbeddingGenerator, ImageFolder
from src.utils.storage_handler import S3Connector
from src.components.nearest_neighbours import Annoy
from timeit import default_timer as timer
from src.utils.common import print_train_time
from src.components.model import NeuralNet
from src.components.trainer import Trainer
from src.logger.logger import logging
from torch.utils.data import DataLoader
from from_root import from_root
from tqdm import tqdm
import torch
import os


class Pipeline:
    def __init__(self):
        self.paths = [
                        "data", "data/raw", "data/splitted", "data/embeddings",
                        "model", "model/benchmark", "model/finetuned"
                    ]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def initiate_data_ingestion(self):
        for folder in self.paths:
            path = os.path.join(from_root(), folder)
            if not os.path.exists(path):
                os.mkdir(folder)
                
        logging.info('Initializing data ingestion...')
        
        dc = DataIngestion()
        dc.run_step()
        
        logging.info("Data ingestion done!\n")

    @staticmethod
    def initiate_data_preprocessing():
        
        logging.info("Initializing data preprocessing (Generating DataLoaders)...")
        
        dp = DataPreprocessing()
        loaders = dp.run_step()
        
        logging.info("Data preprocessing (Generating DataLoaders) done!\n")
        
        return loaders

    @staticmethod
    def initiate_model_architecture():
        return NeuralNet()

    def initiate_model_training(self, loaders, net):
        train_time_start_model = timer()
        trainer = Trainer(loaders, self.device, net)
        trainer.train_model()
        trainer.evaluate(validate=False)
        train_time_end_model = timer()
        
        total_time = (train_time_end_model - train_time_start_model)
        
        logging.info(f"Train time on {self.device}: {total_time:.3f} seconds")
        
        print_train_time(start=train_time_start_model, end=train_time_end_model, 
                        device=self.device)
        
        trainer.save_model_in_pth()

    def generate_embeddings(self, loaders, net):
        logging.info("Initializing embeddings generation...")
        data = ImageFolder(label_map=loaders["valid_data_loader"][1].class_to_idx)
        dataloader = DataLoader(dataset=data, batch_size=32, shuffle=True)
        embeds = EmbeddingGenerator(model=net, device=self.device)

        for batch, values in tqdm(enumerate(dataloader)):
            img, target, link = values
            embedd_response = embeds.run_step(batch, img, target, link)
            logging.info(embedd_response)
            print(embedd_response)
        
        logging.info("Embeddings generations done!!\n")

    @staticmethod
    def create_annoy():
        logging.info("Initializing Ann creation for prediction...")
        
        ann = Annoy()
        ann.run_step()
        
        logging.info("Ann creation done!\n")

    @staticmethod
    def push_artifacts():
        logging.info("Started pushing the artifacts to s3 model registry...")
        connection = S3Connector()
        response = connection.zip_files()
        logging.info("Pushed the artifacts to s3 model registry!\n")
        return response

    def run_pipeline(self):
        logging.info("Starting training pipeline...")
        print("Starting training pipeline...")
        self.initiate_data_ingestion()
        loaders = self.initiate_data_preprocessing()
        net = self.initiate_model_architecture()
        self.initiate_model_training(loaders, net)
        self.generate_embeddings(loaders, net)
        self.create_annoy()
        self.push_artifacts()
        logging.info("Pipeline Run Complete!\n")
        return {"Response": "Pipeline Run Complete"}


if __name__ == "__main__":
    image_search = Pipeline()
    image_search.run_pipeline()