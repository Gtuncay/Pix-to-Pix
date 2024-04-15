import torch
import Utils
from Utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import Config as Config
from Dataset import SchlierenDataset
from Generator import Generator
from Discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

l1_loss_values = []
Results = "Results/MixedL150/"


def train_function(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler, val_loader):
    loop = tqdm (loader, leave = True)
    
    for idx, (x, y) in enumerate (loop):
        x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
        #Training Discriminator
        with torch.cuda.amp.autocast():
            #global y_fake#######################################
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)/2
            
            
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        
        #Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * Config.L1_Lambda
            G_loss = G_fake_loss + L1
            
            ####For printing l1 loss
            l1_loss_values.append(L1.item())

            
           
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
            

def main():
    global val_loader 
    disc = Discriminator(input_channels = 1).to(Config.DEVICE)
    gen = Generator(input_channels = 1).to(Config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr = Config.lr, betas = (0.5, 0.999)) 
    opt_gen = optim.Adam(gen.parameters(), lr = Config.lr, betas = (0.5, 0.999))
    #Loss
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    
    if Config.load_model:
        load_checkpoint(Config.checkpoint_generator, gen, opt_gen, Config.lr)
        load_checkpoint(Config.checkpoint_discriminator, disc, opt_disc, Config.lr)
        
        #Printing directory of checkpoint
        generator_checkpoint_dir = load_checkpoint(Config.checkpoint_generator, gen, opt_gen, Config.lr)
        discriminator_checkpoint_dir = load_checkpoint(Config.checkpoint_discriminator, disc, opt_disc, Config.lr)
       
            
    train_dataset = SchlierenDataset(root_dir_corrupt = Config.TRAIN_DIR, root_dir_target = Config.VAL_DIR) 
    train_dataloader = DataLoader(train_dataset, batch_size = Config.batch_size, shuffle = True, num_workers= Config.num_workers)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = SchlierenDataset(root_dir_target = Config.VAL_DIR, root_dir_corrupt = Config.TRAIN_DIR)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = True) 
    
        
    for epochs in range(Config.num_epochs):
        train_function(disc, gen, train_dataloader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, val_loader) 
            
            
        if Config.save_model and epochs %10 == 0: 
            save_some_examples(gen, val_loader, epochs, folder=Results)
        
        if Config.save_model and epochs %25 == 0:
            save_checkpoint(gen, opt_gen, filename = Results + f"checkpoint_generator_{epochs}.pth.tar")
            save_checkpoint(disc, opt_disc, filename= Results + f"checkpoint_discriminator_{epochs}.pth.tar")    


            
    

if __name__ == "__main__":
    main()
    
        # Plot the L1 loss values
    plt.plot(l1_loss_values, label='L1 Loss')
    plt.xlabel('Iteration')
    plt.ylabel('L1 Loss')
    plt.title('L1 Loss During Training')
    plt.legend()

    # Save the plot to a file
    plt.savefig('l1_loss_plot.png')

    # Show the plot (optional)
    plt.show()
    
