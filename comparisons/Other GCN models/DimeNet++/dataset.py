import numpy as np
import os 
import torch
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info
from file_reader import read_cif 

def endwith(*endstring):
    ends = endstring
    def run(s):
        f = map(s.endswith, ends)
        if True in f:
            return s
    return run
        

def read_data(path):
    subdirs = os.listdir(path)
    graphs = []
    labels = []
    infos = {}
    label_count = 0
    for subdir in subdirs:
        
        pwd = os.path.join(path, subdir)
        files = os.listdir(pwd)
        
        cif_file = os.path.join(pwd, list(filter(endwith('.cif'), files))[0])
        marks_file = os.path.join(pwd, list(filter(endwith('.txt'), files))[0])
        bands_file = os.path.join(pwd, list(filter(endwith('.npy'), files))[0])
        
        graph, hopping_index, d, pd = read_cif(cif_file)
        num_v, num_c = np.loadtxt(marks_file,'int')[0], np.loadtxt(marks_file,'int')[1]
        label = label_count
        
        bands = np.load(bands_file)
        graphs.append(graph)
        labels.append(label)
        label_count+=1
        
        infos[label] = {}
        infos[label]['bands'] = torch.tensor(bands)
        infos[label]['num_v'] = num_v
        infos[label]['num_c'] = num_c
        infos[label]['num_catom'] = torch.where(graph.ndata["Z"] == 6)[0].shape[0]
        infos[label]['hopping_index'] = hopping_index
        infos[label]['d'] = torch.tensor(d).unsqueeze(-1)
        infos[label]['periodic_direction'] = pd
        
    return graphs, torch.tensor(labels), infos


class GGCNNDATASET(DGLDataset):
    """
    The dataset class used to get the dgl graph data from the raw data(crystal structures and bands data)
    """
    def __init__(self, 
                 raw_dir, 
                 save_dir,
                 force_reload = False, 
                 verbose = False):
        super(GGCNNDATASET, self).__init__(name="gnr",
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)
    
    def download(self):
        pass 
    
    def process(self):
        path = self.raw_dir
        self.graphs, self.labels, self.infos  = read_data(path) 
        
    def save(self):
        graph_path = os.path.join(self.save_dir, 'graphs.bin')
        save_graphs(str(graph_path), self.graphs, {'labels': self.labels})
        info_path = os.path.join(self.save_dir, 'infos.bin')
        save_info(info_path, self.infos)
        
    def has_cache(self):
        graph_path = os.path.join(self.save_dir, 'graphs.bin')
        info_path = os.path.join(self.save_dir, 'infos.bin')
        return (os.path.exists(graph_path) and os.path.exists(info_path))
    
    def load(self):
        graphs, label_dict = load_graphs(os.path.join(self.save_dir, 'graphs.bin'))
        infos = load_info(os.path.join(self.save_dir, 'infos.bin'))
        self.graphs = graphs 
        self.labels = label_dict['labels']
        self.infos = infos 
    
    @property
    def get_infos(self):
        return self.infos
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    def get_all(self):
        return self.graphs, self.labels


