import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
import shutil
import random
from glob import glob


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def split_data(self):
        """
         Splits extracted dataset into train/val/test folders.
         Expected input folder format:
             split_dir/
              Cancer/
              Normal/
        Output folder format:
            split_dir/
              train/Cancer, train/Normal
              val/Cancer, val/Normal
              test/Cancer, test/Normal
        """
        random.seed(42)

        source_dir = self.config.split_dir

        # detect class folders (Cancer, Normal)
        class_dirs = [d for d in source_dir.iterdir() if d.is_dir() and d.name not in ["train", "val", "test"]]

        # create output dirs
        for split_name in ["train", "val", "test"]:
            for class_dir in class_dirs:
                out_dir = source_dir / split_name / class_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)

        train_ratio = float(self.config.split_ratio["train"])
        val_ratio = float(self.config.split_ratio["val"])
        test_ratio = float(self.config.split_ratio["test"])

        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("split_ratio must sum to 1.0")

        for class_dir in class_dirs:
            images = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                images.extend(class_dir.glob(ext))

            images = list(images)
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            # remaining goes to test
            n_test = n_total - n_train - n_val
            self.logger.info(f"[{class_dir.name}] total={n_total}, train={n_train}, val={n_val}, test={n_test}")

            train_files = images[:n_train]
            val_files = images[n_train:n_train + n_val]
            test_files = images[n_train + n_val:]

            def copy_files(file_list, split_name):
                for fp in file_list:
                    dst = source_dir / split_name / class_dir.name / fp.name
                    shutil.copy(fp, dst)

            copy_files(train_files, "train")
            copy_files(val_files, "val")
            copy_files(test_files, "test")

        

    
     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


