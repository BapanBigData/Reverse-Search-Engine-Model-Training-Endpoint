from src.entities.config_entities import DataIngestionConfig
from src.logger.logger import logging
from from_root import from_root
import splitfolders
import os


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def download_dir(self):
        """
        params:
        - prefix: pattern to match in s3
        - local: local path to folder in which to place files
        - bucket: s3 bucket with target contents
        - client: initialized s3 client object

        """
        try:
            logging.info("\n====================== Fetching Data ==============================\n")
            print("\n====================== Fetching Data ==============================\n")
            data_path = os.path.join(from_root(), self.config.RAW, self.config.PREFIX)
            
            if not os.path.exists(data_path):
                os.makedirs(data_path)
                print(f"Created directory: {data_path}")
            
            #print(f"Data path for AWS sync: {data_path}")
            logging.info(f"Data path for AWS sync: {data_path}")
            
            os.system(f'aws s3 sync s3://{self.config.BUCKET}/images/ "{data_path}" --no-progress')
            logging.info("\n====================== Fetching Completed ==========================\n")
            print("\n====================== Fetching Completed ==========================\n")

        except Exception as e:
            raise e

    def split_data(self):
        """
        This Method is Responsible for splitting.
        :return:
        """
        try:
            splitfolders.ratio(
                input=os.path.join(self.config.RAW, self.config.PREFIX),
                output=self.config.SPLIT,
                seed=self.config.SEED,
                ratio=self.config.RATIO,
                group_prefix=None, move=False
            )
        except Exception as e:
            raise e

    def run_step(self):
        self.download_dir()
        self.split_data()
        return {"Response": "Completed Data Ingestion"}


# if __name__ == "__main__":
#     paths = ["data", r"data/raw", r"data/splitted", r"data/embeddings",
#             "model", r"model/benchmark", r"model/finetuned"]

#     for folder in paths:
#         path = os.path.join(from_root(), folder)
#         print(path)
#         if not os.path.exists(path):
#             os.mkdir(folder)

#     dc = DataIngestion()
#     print(dc.run_step())