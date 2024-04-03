import numpy as np
import torch.optim as optim
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize  
from matplotlib.image import imread   
from Generator import Generator
from Utils import load_checkpoint
import Config
import random
import pandas as pd


save_path = "Results/MixedOnGauss2/" # Path where metrics are saved

# Validation Dataset
class ValidationDataset(Dataset):
    def __init__(self, root_dir_groundtruth, root_dir_input):
        root_dir_groundtruth = "Test/TestDataset/"
        root_dir_input = "Test/GaussianNoiseLess/"
        self.root_dir_groundtruth = root_dir_groundtruth
        self.root_dir_input = root_dir_input
        
        self.list_files_groundtruth = os.listdir(self.root_dir_groundtruth)
        self.list_files_input = os.listdir(self.root_dir_input)

    def __len__(self):
        return len(self.list_files_input)

    def __getitem__(self, index):
        img_file_groundtruth = self.list_files_groundtruth[index]
        
        # Extracting corrupt files name
        filename, _ = os.path.splitext(img_file_groundtruth)
        
        # For mixed dataset
        for corrupt_file in self.list_files_input:
            if corrupt_file.startswith(filename):
                img_file_input_name = corrupt_file
                break
        
        groundtruth_img_path = os.path.join(self.root_dir_groundtruth, img_file_groundtruth) 
        input_img_path = os.path.join(self.root_dir_input, img_file_input_name)
        
        groundtruth_image = np.array(Image.open(groundtruth_img_path))
        input_image = np.array(Image.open(input_img_path))
        
        augmentations = Config.both_transform(image=input_image, image0=groundtruth_image)
        groundtruth_image, input_image = augmentations["image"], augmentations["image0"]
        
        groundtruth_image = Config.transform_only_input(image=groundtruth_image)["image"]
        input_image = Config.transform_only_input(image=input_image)["image"]
        
        return groundtruth_image, input_image
    
def evaluate(checkpoint_path, val_loader, ssim_values, mse_values, l1_loss_values):
    device = torch.device("cuda")

    gen = Generator(input_channels=1).to(Config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=Config.lr, betas=(0.5, 0.999))
    load_checkpoint(checkpoint_path, gen, opt_gen, Config.lr)

    gen.eval()
    gen.to(device)
    
    ssim_sum = 0.0
    mse_sum = 0.0
    l1_loss_sum = 0.0
    num_samples = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Evaluating {checkpoint_path}"):
            x, y = x.to(device), y.to(device)
            generated_images = gen(x)

            for i in range(len(x)):
                generated_img = generated_images[i].detach().cpu().numpy()
                target_img = y[i].detach().cpu().numpy()
                    
                min_val = np.min(generated_img)
                max_val = np.max(generated_img)
                data_rangegen = max_val - min_val

                ssim_value = ssim(target_img.squeeze(), generated_img.squeeze(), multichannel=False, data_range=data_rangegen) 
                mse_value = mse(target_img, generated_img)
                l1_loss_value = torch.nn.L1Loss()(torch.tensor(target_img), torch.tensor(generated_img)).item()

                ssim_sum += ssim_value
                mse_sum += mse_value
                l1_loss_sum += l1_loss_value
                num_samples += 1
                
                
    ssim_avg = ssim_sum / num_samples
    mse_avg = mse_sum / num_samples
    l1_loss_avg = l1_loss_sum / num_samples
    
    ssim_values.append(ssim_avg)
    mse_values.append(mse_avg)
    l1_loss_values.append(l1_loss_avg)
    print("SSIM list: ", ssim_values)
    print("length ssim: ", len(ssim_values))

    print(f"Checkpoint: {checkpoint_path}")
    print(f"SSIM: {ssim_avg}, MSE: {mse_avg}, L1 Loss: {l1_loss_avg}")

def evaluate_last_checkpoint(last_checkpoint_path, val_loader):
    device = torch.device("cuda")
    
    gen = Generator(input_channels=1).to(Config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=Config.lr, betas=(0.5, 0.999))
    load_checkpoint(last_checkpoint_path, gen, opt_gen, Config.lr)
    
    print(last_checkpoint_path)
    
    for i, (x, y) in enumerate(val_loader):
        random_indices = random.sample(range(len(val_loader)), 7)
        if i in random_indices:
            gen.eval()
            gen.to(device)
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                generated_images = gen(x)

                for j in range(len(x)):
                    generated_img = generated_images[j].detach().cpu().numpy()
                    target_img = y[j].detach().cpu().numpy()
                        

                    # Visualize the output
                    visualize_output(index = i ,y_groundtruth= target_img, y_generator= generated_img, save_path = "Results/MixedOnGauss2/")
              

def visualize_output(index, y_groundtruth, y_generator, save_path):
    save_path = "Results/MixedOnGauss2/"
    if len(y_generator.shape) == 3 and y_generator.shape[0] == 1:
        y_generator = y_generator.squeeze()
    if len(y_groundtruth.shape) == 3 and y_groundtruth.shape[0] == 1:
        y_groundtruth = y_groundtruth.squeeze()
        
    y_groundtruth_resized = resize(y_groundtruth, y_generator.shape[:2], anti_aliasing=True)

    # Calculate global min and max
    global_min = min(np.min(y_generator), np.min(y_groundtruth_resized))
    global_max = max(np.max(y_generator), np.max(y_groundtruth_resized))

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(y_generator, cmap='gray', vmin=global_min, vmax=global_max)
    plt.title('Generator Output')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(y_groundtruth_resized, cmap='gray', vmin=global_min, vmax=global_max)
    plt.title('Ground Truth')
    plt.axis('off') 

    # Calculate absolute difference between output and ground truth
    absolute_difference = np.abs(y_groundtruth_resized - y_generator)

    plt.subplot(1, 3, 3)
    plt.imshow(absolute_difference, cmap='gray')
    plt.title('Difference')
    plt.axis('off')

    # Save the plot
    plot_filename = os.path.join(save_path, f'Visual_Eval{index}.png')
    plt.savefig(plot_filename)
    plt.close()

if __name__ == "__main__":
    val_dataset = ValidationDataset(root_dir_groundtruth= "Test/TestDataset/", root_dir_input= "Test/GaussianNoiseLess/")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    ssim_values_plot = []
    mse_values_plot = []
    l1_loss_values_plot = []
    
    ssim_values = []
    mse_values = []
    l1_loss_values = []    
    
    checkpoint_dir = "Results/MixedL150/"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_generator_') and f.endswith('.pth.tar')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    last_checkpoint_file = checkpoint_files[-1]
    last_checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint_file)

    num_checkpoints = len(checkpoint_files)
    epoch_numbers = range(0, num_checkpoints*25 , 25) ###every 25 epochs one checkpoint created

    evaluate_last_checkpoint(last_checkpoint_path, val_loader)
    
    for epoch_num in epoch_numbers:
        checkpoint_file = f"checkpoint_generator_{epoch_num}.pth.tar"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        evaluate(checkpoint_path, val_loader, ssim_values, mse_values, l1_loss_values)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    epochs = np.arange(len(epoch_numbers)) * 25  #Assuming 25 epochs per checkpoint

    plt.plot(epochs, ssim_values, label='SSIM')
    plt.plot(epochs, mse_values, label='MSE')
    plt.plot(epochs, l1_loss_values, label='L1 Loss')

    plt.xlabel('Epoch Number')
    plt.ylabel('Metrics')
    plt.title('Average Evaluation Metrics over Epochs')
    plt.legend()
    plt.grid(True)
    plt.yticks(np.arange(0, 1.05, 0.05))
    
    plot_filename = os.path.join(save_path, 'evaluation_metrics_plot.png')
    plt.savefig(plot_filename)
    plt.show()
    

    data = {
        'Epoch': epochs,
        'SSIM': ssim_values,
        'MSE': mse_values,
        'L1 Loss': l1_loss_values
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_filename = os.path.join(save_path, 'evaluation_metrics.csv')
    df.to_csv(csv_filename, index=False)