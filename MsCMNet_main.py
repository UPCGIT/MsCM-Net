import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from MsCMNet_model import Unmixing
from utility import load_HSI, hyperVca, load_data, reconstruction_SADloss, SAD
from utility import plotAbundancesGT, plotAbundancesSimple, plotEndmembersAndGT, reconstruct
import time
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import random
import cv2

start_time = time.time()

"""
seed = 3
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
"""


def cos_dist(x1, x2):
    return cosine(x1, x2)


def apply_random_mask(random_mask):
    # Create a copy with the same shape as random_mask
    random_mask_copy = np.copy(random_mask)
    # Gets the index of the part of mask that is False
    false_indices = np.where(random_mask_copy[0] == False)
    # Calculate how many False parts you want to change to True
    num_false = len(false_indices[0])
    num_to_flip = int(num_false * 0.1)
    # Randomly select the index to be True
    indices_to_flip = np.random.choice(num_false, num_to_flip, replace=False)
    # indices_to_flip has two arrays, one for rows and one for columns

    # Sets the selected index position to True(i.e., no mask)
    for index in indices_to_flip:
        row = false_indices[0][index]
        col = false_indices[1][index]

        # Sets all bands of the same pixel to True
        random_mask_copy[:, row, col] = True

    return random_mask_copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasetnames = {'Urban': 'Urban4',
                'Samson': 'Samson',
                'dc': 'DC',
                }
dataset = "dc"  # Replace the data set here
hsi = load_HSI("Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
endmember_number = hsi.abundance_gt.shape[2]
col = hsi.cols
band_number = data.shape[1]

batch_size = 1
EPOCH = 700
num_runs = 1

if dataset == "Samson":
    dloss = 0.001
    drop_out = 0.25
    learning_rate = 0.03
    step_size = 25
    gamma = 0.6
    weight_decay = 1e-3
    min_samples = 13
    eps = 0.001
if dataset == "Urban":
    dloss = 0.001
    drop_out = 0.25
    learning_rate = 0.004
    step_size = 50
    gamma = 0.4
    weight_decay = 1e-4
    min_samples = 13
    eps = 0.001
if dataset == "dc":
    dloss = 0.001
    drop_out = 0.25
    learning_rate = 0.004
    step_size = 50
    gamma = 0.4
    weight_decay = 1e-3
    min_samples = 13
    eps = 0.001

MSE = torch.nn.MSELoss(reduction='mean')

end = []
abu = []
r = []

output_path = 'Results'
method_name = 'MsCM-Net'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

"""
# DBSCAN-VCA
original_HSI1 = torch.from_numpy(data)
original_HSI1 = torch.reshape(original_HSI1.T, (band_number, col, col))
blocks = original_HSI1.unfold(1, 4, 4).unfold(2, 4, 4).reshape(original_HSI1.shape[0], -1, 4, 4).transpose(0, 1)
blocks_new = []
for i in range(blocks.shape[0]):
    if (i + 1) % 100 == 0:
         print("DBSCAN进度：", i + 1, "/", blocks.shape[0])
    # The pixel block is expanded into a two-dimensional matrix
    block_matrix = blocks[i, :, :, :].reshape(blocks.shape[1], -1)
    # Apply the DBSCAN algorithm
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=cos_dist)
    labels = dbscan.fit_predict(block_matrix.T)
    # Removes outliers from the pixel block    
    block_matrix = block_matrix.T[labels != -1, :].T
    blocks_new.append(block_matrix)
blocks = torch.hstack(blocks_new).cpu().numpy()

if dataset == 'Samson':
    sio.savemat('blocks_samson.mat', {'samson': blocks})
elif dataset == 'Urban':
    sio.savemat('blocks_urban.mat', {'urban': blocks})
elif dataset == 'dc':
    sio.savemat('blocks_dc.mat', {'dc': blocks})
"""
if dataset == 'Samson':
    blocks = sio.loadmat('blocks_samson.mat')['samson']
elif dataset == 'Urban':
    blocks = sio.loadmat('blocks_urban.mat')['urban']
elif dataset == 'dc':
    blocks = sio.loadmat('blocks_dc.mat')['dc']

for run in range(1, num_runs + 1):
    print('Start training!', 'run:', run)
    abundance_GT = torch.from_numpy(hsi.abundance_gt)
    abundance_GT = torch.reshape(abundance_GT, (col * col, endmember_number))
    original_HSI = torch.from_numpy(data)
    original_HSI = torch.reshape(original_HSI.T, (band_number, col, col))
    abundance_GT = torch.reshape(abundance_GT.T, (endmember_number, col, col))

    image = np.array(original_HSI)
    """
    # Calculate the cosine similarity to get similarity matrix
    # The closer the cosine distance is to 1, the greater the similarity value. However, in this version of the code, 
    # the cosine distance is subtracted by 1 and multiplied by 1000, initially to better observe the difference between similarities.
    # At this point in the code, the smaller the value in the similarity matrix, the more similar it is.  
    # You can operate the similarity matrix as follows: similarity_matrix = 1 - similarity_matrix/1000;   
    # This corresponds to the explanation in the paper. It's the same in theory.
    # The two are only different in operation, and there is no essential difference.
    similarity_matrix = np.zeros_like(image)
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            center_pixel = image[:, i, j]
            neighbors = []
            # Gets the values of the four surrounding pixels
            if i > 0:
                neighbors.append(image[:, i - 1, j])
            if i < image.shape[1] - 1:
                neighbors.append(image[:, i + 1, j])
            if j > 0:
                neighbors.append(image[:, i, j - 1])
            if j < image.shape[2] - 1:
                neighbors.append(image[:, i, j + 1])
            a = np.array(neighbors)
            if neighbors:
                similarities = cosine_similarity(center_pixel.reshape(1, -1), np.array(neighbors))
                similarity_matrix[:, i, j] = (1 - np.mean(similarities)) * 1000
    if dataset == 'Samson':
        sio.savemat('similarity_samson.mat', {'samson': similarity_matrix})
    elif dataset == 'Urban':
        sio.savemat('similarity_urban.mat', {'urban': similarity_matrix})
    elif dataset == 'dc':
        sio.savemat('similarity_dc.mat', {'dc': similarity_matrix})
    """

    if dataset == 'Samson':
        similarity_matrix = sio.loadmat('similarity_samson.mat')['samson']
    elif dataset == 'Urban':
        similarity_matrix = sio.loadmat('similarity_urban.mat')['urban']
    elif dataset == 'dc':
        similarity_matrix = sio.loadmat('similarity_dc.mat')['dc']
    flattened_matrix = similarity_matrix.flatten()
    normalized_matrix = flattened_matrix.astype(np.uint16)
    # OTSU
    threshold, a = cv2.threshold(normalized_matrix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clean_pixels = np.where(similarity_matrix < threshold)
    # Create mask
    mask = np.zeros_like(image, dtype=bool)
    mask[:, clean_pixels[1], clean_pixels[2]] = True
    mask_image = mask * image

    endmembers, _, _ = hyperVca(blocks, endmember_number, datasetnames[dataset])  # DBSCAN-VCA
    VCA_endmember = torch.from_numpy(endmembers)
    GT_endmember = hsi.gt.T
    endmember_init = VCA_endmember.unsqueeze(2).unsqueeze(3).float()

    # load data
    train_dataset = load_data(img=original_HSI, transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    net = Unmixing(band_number, endmember_number, drop_out, col).cuda()

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name

    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name


    def weights_init(m):
        nn.init.kaiming_normal_(net.layer1[0].weight.data)
        nn.init.kaiming_normal_(net.layer1[4].weight.data)
        nn.init.kaiming_normal_(net.layer1[8].weight.data)

        nn.init.kaiming_normal_(net.layer2[0].weight.data)
        nn.init.kaiming_normal_(net.layer2[4].weight.data)
        nn.init.kaiming_normal_(net.layer2[8].weight.data)

        nn.init.kaiming_normal_(net.layer3[0].weight.data)
        nn.init.kaiming_normal_(net.layer3[4].weight.data)
        nn.init.kaiming_normal_(net.layer3[8].weight.data)


    net.apply(weights_init)

    # decoder weight init by VCA
    model_dict = net.state_dict()
    model_dict["decoderlayer4.0.weight"] = endmember_init
    model_dict["decoderlayer5.0.weight"] = endmember_init
    model_dict["decoderlayer6.0.weight"] = endmember_init

    net.load_state_dict(model_dict)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    y = original_HSI.unsqueeze(0).to(device)
    y2 = nn.MaxPool2d(2, 2, ceil_mode=True)(y).to(device)
    y3 = nn.MaxPool2d(4, 4, ceil_mode=True)(y).to(device)

    for epoch in range(EPOCH):
        for i, x in enumerate(train_loader):
            x = x.cuda()
            net.train().cuda()

            random_mask = torch.from_numpy(apply_random_mask(mask)).cuda()
            x = (x.squeeze() * random_mask).unsqueeze(0)

            en_abundance, reconstruction_result, en_abundance2, reconstruction_result2, en_abundance3, reconstruction_result3, x2, x3 = net(
                x)

            abundanceLoss = reconstruction_SADloss(y, reconstruction_result)
            abundanceLoss2 = reconstruction_SADloss(y2, reconstruction_result2)
            abundanceLoss3 = reconstruction_SADloss(y3, reconstruction_result3)

            loss3 = torch.sum(torch.pow(torch.abs(en_abundance) + 1e-8, 0.5))
            loss2 = torch.sum(torch.pow(torch.abs(en_abundance2) + 1e-8, 0.5))
            loss1 = torch.sum(torch.pow(torch.abs(en_abundance3) + 1e-8, 0.5))

            ALoss = abundanceLoss + abundanceLoss2 + abundanceLoss3
            DLoss = 1e-4 * loss3
            total_loss = ALoss + dloss * DLoss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                """print(ELoss.cpu().data.numpy())"""
                print("Epoch:", epoch, "| loss: %.4f" % total_loss.cpu().data.numpy())

    net.eval()

    en_abundance, reconstruction_result, en_abundance2, reconstruction_result2, en_abundance3, reconstruction_result3, x2, x3 = net(
        x)
    en_abundance = torch.squeeze(en_abundance)

    en_abundance = torch.reshape(en_abundance, [endmember_number, col * col])
    en_abundance = en_abundance.T
    en_abundance = torch.reshape(en_abundance, [col, col, endmember_number])
    abundance_GT = torch.reshape(abundance_GT, [endmember_number, col * col])
    abundance_GT = abundance_GT.T
    abundance_GT = torch.reshape(abundance_GT, [col, col, endmember_number])
    en_abundance = en_abundance.cpu().detach().numpy()
    abundance_GT = abundance_GT.cpu().detach().numpy()

    endmember_hat = net.state_dict()["decoderlayer4.0.weight"].cpu().numpy()
    endmember_hat = np.squeeze(endmember_hat)
    endmember_hat = endmember_hat.T

    GT_endmember = GT_endmember.T
    y_hat = reconstruct(en_abundance, endmember_hat)
    RE = np.sqrt(np.mean(np.mean((y_hat - data) ** 2, axis=1)))
    r.append(RE)

    sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'A': en_abundance,
                                                                              'E': endmember_hat})

    plotAbundancesSimple(en_abundance, abundance_GT, abundance_path, abu)
    plotEndmembersAndGT(endmember_hat, GT_endmember, endmember_path, end)

    torch.cuda.empty_cache()

    print('-' * 70)
end_time = time.time()
end = np.reshape(end, (-1, endmember_number + 1))
abu = np.reshape(abu, (-1, endmember_number + 1))
dt = pd.DataFrame(end)
dt2 = pd.DataFrame(abu)
dt3 = pd.DataFrame(r)
# SAD and mSAD results of each endmember for multiple runs are saved to csv files
dt.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' +
          'SAD and mSAD results for multiple runs.csv')
# RMSE and mRMSE results of each abundance for multiple runs are saved to csv files
dt2.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' +
           'RMSE and mRMSE results for multiple runs.csv')
# RE results of each abundance for multiple runs are saved to csv files
dt3.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'RE results for multiple runs.csv')
# abundance GT
abundanceGT_path = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'Abundance GT'
plotAbundancesGT(hsi.abundance_gt, abundanceGT_path)
print('Running time:', end_time - start_time, 's')
