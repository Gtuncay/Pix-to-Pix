import torch
import Config as Config
from torchvision.utils import save_image
import os
Results = "Results/MixedL150/GeneratorOutput/" #Change here depending on what kind of corruption you are fixing
from Dataset import SchlierenDataset
flag = False


def save_some_examples(gen, val_loader, epoch, folder): 
    
    global flag
    flag = True
    
    x, y = next(iter(val_loader))
    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)

        #plot min and max
        # min_value = torch.min(y_fake).item()
        # max_value = torch.min(y_fake).item()

        # print("Max: ", max_value)
        # print("Min: ", min_value)
        
        y_fake = y_fake * 0.10035 + 0.509  
   
        
        save_image(y_fake , folder + f"/GeneratorOutput/y_gen_MixedL150_{epoch}.png") 
        save_image(x * 0.10035 + 0.509, folder + f"/input_{epoch}.png") 
        
        GeneratedImagePath = os.path.join(Results + f"y_gen_{epoch}.png")
        
        
    gen.train()
    return GeneratedImagePath
    

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
    # Extract directory name from the filepath
    checkpoint_dir = os.path.dirname(checkpoint_file)
    return checkpoint_dir
