# @dataset{Food Recognition 2022,
# 	author={AIcrowd},
# 	title={Food Recognition 2022},
# 	year={2022},
# 	url={https://www.kaggle.com/datasets/awsaf49/food-recognition-2022-dataset}
# }
import numpy as np
import os
import tarfile
import csv
import json

tar_directory = "/"
tar_find = os.path.join(tar_directory, "food-recognition-DatasetNinja.tar")
test = "./food-recognition-DatasetNinja.tar"


testing_csv_path = "testing_files.csv"
validation_csv_path = "validation_files.csv"
training_csv_path = "training_files.csv"

meta_dir = "./../meta.json"

testing_data = []  
validation_data = []  
training_data = []

class DataInfo: 
    """
    This class just contains functions about the meta data
    """

    def populate_meta_data(): 
        """
        Populates the meta data json into the meta data folder
        """
        with tarfile.open(test, "r") as tar:
            tar_contents = tar.getnames()
            print("Contents of the tar file:")
            for name in tar_contents:
                print(name)
            json_file = next((name for name in tar_contents if name.endswith(".json")), None)
            if json_file:
                print(f"Extracting {json_file}...")
                tar.extract(json_file, path=meta_dir)
                extracted_path = os.path.join(meta_dir, json_file)
                print(f"File extracted to: {extracted_path}")
            else:
                print("No JSON file found in the tar archive.")

    def num_classes(): 
        """
        Returns the number of classes in the dataset
        """
        path = "./../../meta.json"
        with open(path, "r") as file:
            data = json.load(file)

        if "classes" in data:
            classes = data["classes"]
            num_classes = len(classes)
        else:
            print("No 'classes' key found in the JSON file.")
        print(num_classes)
        return num_classes


    def return_classes(): 
        """
        Returns the a set of the unique classes in the dataset
        """
        path = "../Image_Model/meta.json"
        unique_labels = []
        with open(meta_dir, "r") as file:
            data = json.load(file)
        for i in data["classes"]: 
            unique_labels.append(i["title"])
        return list(set(unique_labels))
    
    