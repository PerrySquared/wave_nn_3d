from tqdm import tqdm
from data import WaveDataset
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
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
    
def recursive_search(comb, current_column, current_row, current_height):
    
    possibilities = 0
    row_len, column_len, height_len = comb.shape
    # print(comb.shape)
    if current_column > 0:
        if comb[current_column - 1, current_row, current_height] == 2:
            return 1
        if comb[current_column - 1, current_row, current_height] == 1:
            possibilities += 1
            comb[current_column, current_row, current_height] = 0
            return recursive_search(comb, current_column - 1, current_row, current_height)
        
    if current_column < column_len - 1:
        if comb[current_column + 1, current_row, current_height] == 2:
            return 1 
        if comb[current_column + 1, current_row, current_height] == 1:
            possibilities += 1  
            comb[current_column, current_row, current_height] = 0
            return recursive_search(comb, current_column+1, current_row, current_height)     
        
        
    if current_row > 0:
        if comb[current_column, current_row - 1, current_height] == 2:
            return 1   
        if comb[current_column, current_row - 1, current_height] == 1:
            possibilities += 1  
            comb[current_column, current_row, current_height] = 0
            return recursive_search(comb, current_column, current_row-1, current_height)      
        
    if current_row < row_len - 1:
        if comb[current_column, current_row + 1, current_height] == 2:
            return 1
        if comb[current_column, current_row + 1, current_height] == 1:
            possibilities += 1  
            comb[current_column, current_row, current_height] = 0
            return recursive_search(comb, current_column, current_row+1, current_height)
        
        
    if current_height > 0:
        if comb[current_column, current_row, current_height - 1] == 2:
            return 1    
        if comb[current_column, current_row, current_height - 1] == 1:
            possibilities += 1
            comb[current_column, current_row, current_height] = 0
            return recursive_search(comb, current_column, current_row, current_height-1)    
        
    if current_height < height_len - 1:
        if comb[current_column, current_row, current_height + 1] == 2:
            return 1                
        if comb[current_column, current_row, current_height + 1] == 1:
            possibilities += 1
            comb[current_column, current_row, current_height] = 0
            return recursive_search(comb, current_column, current_row, current_height+1)
    
    
    if possibilities == 0:
        return 0


def defection_check(source, predict):

    # print(source.shape, predict.shape)
    source, predict = source.cpu().squeeze().permute(1,2,0), predict.cpu().permute(1,2,0)
    start_row, start_column, start_height = 0, 0, 0

    predict = torch.where(predict > -5, 1.0, 0.0)

    row_len, column_len,height_len = source.shape
    # print(height_len, row_len, column_len)
    # print(source.shape)
    for height in range(height_len):
        for row in range(row_len):
            for column in range(column_len):
                # print(height)
                # print(source.shape)
                
                if source[column, row, height] > 0.666 and source[column, row, height] < 0.668:
                    start_row = row
                    start_column = column
                    start_height = height
    
    comb = source + predict
    current_row, current_column, current_height = start_row, start_column, start_height
    
    defect = recursive_search(comb, current_column, current_row, current_height)
     
    return defect  

if __name__ == '__main__':
    
    set_seed()
    
    data_size = 50
    batch_size = 64

    data_test = pd.read_csv("data.csv", nrows=data_size)

    dataset_test = WaveDataset(data_test)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    print(device)
    
    aux_params = dict(
        dropout = 0.70,
        classes=1,
        activation="tanh"
    )
    
    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                     in_channels=3, decoder_attention_type=None, classes=3).to(device)

    
    
    checkpoint = torch.load('best_model.pth')
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    name = 0

    model.to(device)
    model.eval()
    valid_accuracy = np.array([])
    path_lengts_avg_mask = np.array([])
    path_lengts_avg_pred = np.array([])
    
    with torch.no_grad():
        
        for source, mask in tqdm(dataloader_test):
            
            source, mask = source.to(device), mask.to(device)
            out = model(source)
            out = torch.squeeze(out)
       
            for i in range(len(mask)):
                
                name += 1
                
                torch.save(out[i].clone().detach(), os.path.join("./data_out_trained/tensors", str(name)))

                # path_lengts_avg_mask = np.append(path_lengts_avg_mask, mask[i].cpu().sum())
                # path_lengts_avg_pred = np.append(path_lengts_avg_pred, torch.where(out[i] > -5, 1.0, 0.0).cpu().sum())
                # # print(source.shape, out.shape)
                # if(mask[i].sum() > 0):     
                #     valid_accuracy = np.append(valid_accuracy, defection_check(source[i], out[i]))
                
                # code below was used to generate images, but commented for the sake of benchmarking

                
                
                source = torch.squeeze(source).cpu()    
                mask = torch.squeeze(mask).cpu()
                
                
                image = out.cpu()
                image = torch.where(image > 0, 1.0, 0.0).permute(0,2,3,1)
                # print(source.shape,mask.shape,image.shape)
                
                # print(set(mask[i]))
                # print(set(image[i]))
                # print(set(source[i]))
                
                
                fig = plt.figure().add_subplot(projection='3d')
                # ax=fig.gca()
                # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.1, 1]))
                # print(mask[i].permute(1,2,0))
                
                # torch.set_printoptions(profile="full")
                # print(mask)
                fig.voxels(mask[i].permute(1,2,0))
                # plt.imshow(mask[i] + source[i])
                plt.savefig(os.path.join("data_out_trained/images", str(name) + "_mask"), dpi=300)
                
                fig = plt.figure().add_subplot(projection='3d')
                # plt.imshow(image[i] + source[i])
                fig.voxels(image[i])
                plt.savefig(os.path.join("data_out_trained/images", str(name) + "_predicted"), dpi=300)
                
                fig = plt.figure().add_subplot(projection='3d')
                fig.voxels(source[i].permute(1,2,0), facecolors='red')
                # plt.imshow(mask[i] + source[i])
                plt.savefig(os.path.join("data_out_trained/images", str(name) + "_source"), dpi=300)
                plt.close()  
                
# print(valid_accuracy.mean())   
# print(path_lengts_avg_mask.mean())           
# print(path_lengts_avg_pred.mean())    
