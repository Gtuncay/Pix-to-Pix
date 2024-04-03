from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import Config as Config
from torchvision.utils import save_image
import Utils
import csv

class SchlierenDataset(Dataset):
    def __init__(self, root_dir_target, root_dir_corrupt):
        
        root_dir_target = "Training/CutDataset/CutDataset/" #DIRECTORY OF UNCORRUPTED IMAGES
        root_dir_corrupt = "Training/CorruptedDataset/MixedDataset/" #DIRECTORY OF CORRUPTED IMAGES
        #Target data root
        self.root_dir_target = root_dir_target
        #Corrupt data root
        self.root_dir_corrupt = root_dir_corrupt
                
        self.list_files_target = os.listdir(self.root_dir_target)
        self.list_files_corrupt = os.listdir(self.root_dir_corrupt)

        
    def __len__(self): 
        return len(self.list_files_corrupt)
    
    def __getitem__(self, index):
        img_file_target = self.list_files_target[index]     
        if Utils.flag: 
             
            #Saving groundtruths in a csv file
            Groundtruthpath = os.path.join("Training/CutDataset/CutDataset/" + img_file_target)#GT path
            print("Groundtruthpath Dataset: ", Groundtruthpath)
            
            with open('Results/MixedL150/groundtruth_paths.csv', mode='a', newline='') as file:
                writer = csv.writer(file)

                # Check if the file is empty and write header if it is
                if os.stat('Results/MixedL150/groundtruth_paths.csv').st_size == 0:
                    writer.writerow(['Groundtruthpath'])

                writer.writerow([Groundtruthpath])

            Utils.flag = False
           
        
            
        # Extracting corrupt files
        filename, _ = os.path.splitext(img_file_target)
       
        for corrupt_file in self.list_files_corrupt:
            if corrupt_file.startswith(filename):
                img_file_corrupt = corrupt_file
                break

        
        targetimg_path = os.path.join(self.root_dir_target, img_file_target) 
        corruptedimg_path = os.path.join(self.root_dir_corrupt, img_file_corrupt)    
        
        
        corrupted_image = np.array(Image.open(corruptedimg_path))
        target_img = np.array(Image.open(targetimg_path)) 
        
        
        input_image = corrupted_image 
        target_image = target_img 
        
        augmentations = Config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"] 
        
        #print("After augmentations - Input shape:", input_image.shape)
        #print("After augmentations - Target shape:", target_image.shape)
        
        
        input_image = Config.transform_only_input(image = input_image)["image"]
        target_image = Config.transform_only_input(image = target_image)["image"]

        return input_image, target_image


        
        
if __name__ == "__main__":
    dataset = SchlierenDataset("Training/CorruptedDataset/MixedDataset/") #DIRECTORY OF CORRUPTED IMAGES
    loader = DataLoader(dataset, batch_size=10)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()