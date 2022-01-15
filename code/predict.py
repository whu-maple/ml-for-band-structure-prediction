import numpy as np
import torch
from core.file_reader import read_cif
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
from math import ceil
from core.model import WHOLEMODEL

def predict(model, target_cif):
    
    graph, t, d, pd = read_cif(target_cif)

    num_catoms = len(torch.where(graph.ndata['species']==6)[0])
    
    o, h = model(graph, torch.tensor(t, dtype = torch.int64), torch.tensor(d).view(-1,1))    

    h0,h1 = torch.chunk(h,2,0)
    h0 = torch.reshape(h0,[num_catoms,num_catoms])
    o = torch.reshape(o,[-1])
    h0 = h0 - torch.diag(torch.diag(h0)) + torch.diag(o)
    h1 = torch.reshape(h1,[num_catoms,num_catoms])
 
    return h0.detach().numpy(), h1.detach().numpy()

def get_bands(h0,h1):
    Energy_hr=np.array([])  
    H_size=h0.shape[0]
    n = 26
    for k in range(n):
        H = np.exp(-1j*np.pi*0.04*k)*h1.T + h0 + np.exp(1j*np.pi*0.04*k)*h1 
        E = np.linalg.eigvals(H)
        E = np.sort(E.real)    
        Energy_hr=np.append(Energy_hr,E)
    Energy_hr=Energy_hr.reshape(H_size,-1,order='F')
    return Energy_hr.T

def plot_bands(pbands):
    
    plt.figure()
    x = np.linspace(0,25,26)
    
    for i in range(pbands.shape[1]):
        plt.plot(x, pbands[:,i], color="r", ls="-", label = "predicted bands")
        
    plt.xlim(0,25)
    plt.ylim(-4,4)
    plt.show()
    plt.xticks([0,25],['G','Z']) 
     
def main(pkl_file, cif_file):
    
    """
    Here we show the example of predicting the band structure for 26-atom-wide armchair
    graphene nanoribbon with H saturated edges. Just to change 'predict_structure_cif_path'
    and 'references' to see the predictions for other systems and their comparisons to ab initio ones
    """
    
    # initial the model
    model = WHOLEMODEL(
                            gnn_dim_list = [30, 27, 24, 20],
                            gnn_head_list = [5, 3, 1],
                            onsite_dim_list = [20, 15, 10, 5, 1],
                            hopping_dim_list1 = [20, 40, 35, 40],
                            hopping_dim_list2 = [80, 20, 15, 7, 1],
                            expander_bessel_dim = 40,
                            expander_bessel_cutoff = 20,
                           )
    
    # lode the trained network parameters
    checkpoint = torch.load(pkl_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # the intermediate Hamiltonian prediction
    H0, H1 = predict(model, cif_file)

    # predict and plot band structure
    plot_bands(get_bands(H0,H1))
    
if __name__ == '__main__':
    main(pkl_file = './results/10-fold cross-validation/Fold6.pkl',  cif_file = '../data/data set/10-fold cross-validation/Fold6/raw/test/HJ_2W7_2W9/7-2-9-2-HJ.cif')