from tqdm import tqdm
from data import WaveDataset
from torch.utils.data import Dataset, DataLoader
from  sklearn.model_selection import train_test_split
import torch
import os
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smpl



def set_seed(seed=777):
    # Set seed for every random generator that used in project
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def dice_coef(y_true, y_pred, thr=0.5, dim=(1,2,3), epsilon=0.001):
    # print(y_true.shape, y_pred.shape)
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    # print(y_true.shape, y_pred.shape)
    inter = (y_true*y_pred).sum(dim=dim)
    # print(inter.shape)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    # print(den.shape)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean()
    # print(dice.shape, dice)
    return dice


class SaveBestModel:

    def __init__(
        self, best_accuracy = 0
    ):
        self.best_accuracy = best_accuracy
        
    def __call__(
        self, epoch, current_accuracy, model, optimizer
    ):
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            torch.save({
                'epoch': epoch+1,
                'accuracy':self.best_accuracy,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'best_model.pth')


if __name__ == '__main__':
    
    set_seed()


    data_train, data_test = train_test_split(pd.read_csv("data.csv", nrows=60000), train_size=0.9, test_size=0.1, shuffle=False)

    save_best_model = SaveBestModel()

    dataset_train = WaveDataset(data_train)
    dataset_test = WaveDataset(data_test)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=64, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
    aux_params = dict(
        dropout = 0.70,
        activation="tanh"
    )
    
    model = smp.UnetPlusPlus(encoder_name="vgg16", encoder_depth=5, encoder_weights="imagenet", decoder_attention_type=None,
                     in_channels=3, decoder_channels=(256, 128, 64, 32, 16), classes=3, activation=None).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7000, eta_min=1e-6)
    DiceLoss = smpl.DiceLoss(mode='multilabel')
    
    # checkpoint = torch.load('best_model.pth')
    # epoch = checkpoint['epoch']
    # accuracy = checkpoint['accuracy']
    # model.load_state_dict(checkpoint['model_state_dict'])

    def criterion(input_masks, target_masks):
        return DiceLoss(input_masks, target_masks)


    metric = np.zeros((0, 2))
    image_number = 0

    epochs = 40

    for i in range(epochs):
        train_loss = np.array([])
        valid_accuracy = np.array([])

        model.train()

        for source, mask in tqdm(dataloader_train):
            optimizer.zero_grad()

            source, mask = source.to(device), mask.to(device)
            # print("source.shape",source.shape)
            # print("mask.shape",mask.shape)
            out = model(source)
            # print("out.shape",out.shape)
            # print("mask.shape",mask.shape)
            # mask = mask.unsqueeze(dim=1).float()
            # print(mask.shape)
            loss = criterion(out.float(), mask)

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss = np.append(train_loss, loss.item())

        model.eval()

        with torch.no_grad():
            
            for source, mask in tqdm(dataloader_test):
                source, mask = source.to(device), mask.to(device)
                # print("source.shape",source.shape)
                # print("mask.shape",mask.shape)                
                out = model(source)
                # print("out.shape",out.shape)
                valid_accuracy = np.append(valid_accuracy, dice_coef(mask, out.float()).cpu().detach().numpy())
                
                
            #    commented part below was used to ckeck progress of training, works only with validation batch size = 1

            #     image = torch.squeeze(out[0]).cpu()
            #     temp_mask = torch.squeeze(mask).cpu()
                
            # image_number+=1
            # image = torch.where(image > 0.5, 1.0, 0.0)
            
            # plt.imshow(temp_mask)
            # plt.savefig(os.path.join("./data_out", str(image_number) + "_mask"), dpi=300)
            # plt.imshow(image)
            # plt.savefig(os.path.join("./data_out", str(image_number)), dpi=300)       
            # plt.close()  
        

        save_best_model(
            i,  np.mean(valid_accuracy), model, optimizer
        )
        
        
        metric = np.append(metric, [[np.mean(train_loss), np.mean(valid_accuracy)]], axis=0)

        print(
            f'\n---> {i+1}\033[95m LR:\033[0m {optimizer.param_groups[0]["lr"]:3e}' +
            f'\n|\033[94m Loss_train:\033[0m {metric[-1, 0]:.5}' +
            f'\n|\033[96m Valid_accuracy:\033[92m {metric[-1, 1]:.5}' +
            '\n---------------------------\033[0m')
        
    loss = metric[:,0]
    accuracy = metric [:,1]
    plt.plot([i for i in range(epochs)], loss)
    plt.show()
    plt.plot([i for i in range(epochs)], accuracy)
    plt.show()

