import numpy as np
import torch
from file_reader import read_cif
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
from math import ceil

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

def compare_bands(pbands,rbands):
    
    plt.figure()
    x = np.linspace(0,25,26)
    
    for i in range(rbands.shape[1]):
        plt.plot(x, rbands[:,i], color="b", label = "ab-initio bands")
        
    for i in range(pbands.shape[1]):
        plt.plot(x, pbands[:,i], color="r", ls="--", label = "predicted bands")
        
    plt.xlim(0,25)
    plt.ylim(-4,4)
    plt.show()
    plt.xticks([0,25],['G','Z']) 
     
def main():
    
    """
    Here we show the example of predicting the band structure for 26-atom-wide armchair
    graphene nanoribbon with H saturated edges. Just to change 'predict_structure_cif_path'
    and 'references' to see the predictions for other systems and their comparisons to ab initio ones
    """

    # lode the trained network parameters
    model = torch.load('./trained_model.pkl')

    # choose the cif file of the desired system from the test set
    predict_structure_cif_path = './dataset/testset/raw/AGNR_H_26/AGNR_H_26.cif'

    # the intermediate Hamiltonian prediction
    H0, H1 = predict(model, predict_structure_cif_path)

    # get the predicted band structure and compare it with the ab initio references
    references = "./ab initio band structure/1 H-AGNR/bands_AGNR_H_26.npy"
    compare_bands(get_bands(H0, H1), np.load(references))
    
if __name__ == '__main__':
    main()