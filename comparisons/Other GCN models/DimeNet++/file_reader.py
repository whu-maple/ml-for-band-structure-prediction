import numpy as np
import dgl
import torch
from pymatgen.core.structure import Structure


def read_cif(cif_file, r_neighborhood = 4.0):
    
    crystal = Structure.from_file(cif_file)
    lattice = crystal._lattice._matrix
    atomic_numbers = np.array(crystal.atomic_numbers)
    coords = crystal.cart_coords
    num_atom = coords.shape[0]
    
    all_nbrs = crystal.get_all_neighbors(r_neighborhood, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    
    nbr_r = [np.array(list(map(lambda x: x[1], nbr[:]))) for nbr in all_nbrs]
    nbr_id = [np.array(list(map(lambda x: x[2], nbr[:]))) for nbr in all_nbrs]
    nbr_image = [np.array(list(map(lambda x: x[3], nbr[:]))) for nbr in all_nbrs]
    
    src = []
    dst = []
    edge_r = []
    edge_o = []
    for i in range(num_atom):
        num_nbr = len(nbr_id[i])
        for j in range(num_nbr):
            src.append(int(i))
            dst.append(int(nbr_id[i][j]))
            edge_r.append(nbr_r[i][j])
            o = coords[int(i)] - (coords[int(nbr_id[i][j])] + np.matmul(nbr_image[i][j], lattice))
            edge_o.append(torch.tensor(o).reshape([1, -1]))
            
    graph = dgl.graph((src,dst))
    edge_o = torch.cat(edge_o,dim=0)
    
    graph.ndata["species"] = torch.tensor(atomic_numbers)
    graph.ndata["Z"] = torch.tensor(atomic_numbers)
    graph.edata["o"] = edge_o
    graph.edata["d"] = torch.tensor(edge_r)
    
    carbon_coords = coords[np.where(atomic_numbers==6)]
    num_catoms = carbon_coords.shape[0]
    
    nbr_image = [np.array(list(map(lambda x: x[3], nbr[:]))) for nbr in all_nbrs]
    _, periodic_direction = np.where(np.unique(np.abs(np.concatenate(nbr_image)),axis=0)==1) 
    periodic_direction = periodic_direction[0] #0:x 1:y 2:z

    images = np.zeros([2,3])
    images[1, periodic_direction] = 1
        
    hopping_index = []
    for i in images:
        for j in range(num_catoms):
            for k in range(num_catoms):
                tmp = np.array([j, k])
                hopping_index.append(tmp)
    hopping_index = np.array(hopping_index)
    
    d = []
    for i in images:
        for j in range(num_catoms):
            atom1_coords = carbon_coords[j]
            for k in range(num_catoms):
                atom2_coords = carbon_coords[k] + np.matmul(i,lattice)
                distance = np.linalg.norm(atom1_coords-atom2_coords)
                if not distance == 0.:
                    d.append(distance)
                else: # distance being 0 means the same atom
                    d.append(1.0) 
                    #this value make no difference 
                    #because it will be replaced by onsite terms when constructing Hamiltonian matrics
    d = np.array(d)
    return graph, hopping_index, d, periodic_direction


