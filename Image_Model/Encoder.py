import json

class Image_Encoder: 
    def decoder(): 
        with open("../../encodings.json", "r") as f:
            label_to_index = json.load(f)
            index_to_label = {v: k for k, v in label_to_index.items()}
        return index_to_label
    
    def encoder(): 
        with open("../../encodings.json", "r") as f: 
            label_to_index = json.load(f)
        return label_to_index
