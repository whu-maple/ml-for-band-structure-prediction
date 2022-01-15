import numpy as np
import torch
import torch.nn.functional as F
import dgl
import random
import os
from core.model import WHOLEMODEL
from core.dataset import GGCNNDATASET
from dgl.dataloading import GraphDataLoader
from math import ceil
import json
torch.set_default_tensor_type(torch.DoubleTensor)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def read_infos(labels,infos):
    
    label_list = [i for i in labels.numpy()]
    hopping_index_list = [infos[i]['hopping_index'] for i in label_list]
    c_atom_list = [infos[i]['num_catom'] for i in label_list]
    hopping_index_batch = [hopping_index_list[i] + sum(c_atom_list[:i]) for i in range(len(hopping_index_list))]
    hopping_index_batch = [torch.tensor(i, dtype=torch.int64) for i in hopping_index_batch]
    d_list = [infos[i]['d'] for i in label_list]
    return torch.cat(hopping_index_batch), torch.cat(d_list)

def get_eikr(kpoint = 26):
    """
    in our work, the studied systems are 1-D and the k sampling is 1*1*50 along Gamma to Z
    so 1*1*26 along [0,0,0] to [0,0,1/2] in units of 2pi/az
    this function is used to obtain the exp(i K dot R) terms 
    for H(R) (R = [0,0,0], [0,0,1], and [0,0,-1]) in units of [ax, ay, az])
    which will be used when computing the H(K) = ∑ exp(i K dot R) * H(R)  
    """
    k = np.linspace(0, np.pi, kpoint)
    K = torch.tensor(k)
    eikr = torch.tensor([])
    for i in range(kpoint):
        tmp = torch.cat([torch.reshape(torch.exp(1j*K[i]*-1),[1,-1]),
                         torch.reshape(torch.exp(1j*K[i]*0),[1,-1]),
                         torch.reshape(torch.exp(1j*K[i]*1),[1,-1])],1)
        eikr = torch.cat([eikr,tmp],0)
    
    eikr = eikr.reshape([kpoint,3,1]) 
    
    return torch.chunk(eikr,3,1)

    
def compute_loss(output, labels, infos, eikr_m1, eikr_0, eikr_1):
    
    label_list  = [i for i in labels.numpy()]
    num_catoms  = [infos[i]['num_catom'] for i in label_list]
    num_c       = [infos[i]['num_c'] for i in label_list]
    num_v       = [infos[i]['num_v'] for i in label_list]
    references  = [infos[i]['bands'] for i in label_list]
    num_matelements = [2*i**2 for i in num_catoms] 
    
    loss = []
    o, h = output
    o_s = torch.split(o, num_catoms)
    h_s = torch.split(h, num_matelements)
    for onsite, hopping, catoms, cbands, vbands, rbands in zip(o_s, h_s, num_catoms, num_c, num_v, references):
        h0,h1 = torch.chunk(hopping,2,0)
        h0 = torch.reshape(h0,[catoms,catoms])
        on = torch.reshape(onsite,[-1])
        h0 = h0 - torch.diag(torch.diag(h0)) + torch.diag(on)
        h1 = torch.reshape(h1,[catoms,catoms])
        hm1 = h1.t()
        h0 = h0.expand(26, catoms,catoms)
        h1 = h1.expand(26, catoms,catoms)
        hm1 = hm1.expand(26, catoms,catoms)
        hk = eikr_m1*hm1 + eikr_0*h0 + eikr_1*h1
        w, v = torch.linalg.eigh(hk)
        e_r = w[:,ceil(catoms/2) - vbands:ceil(catoms/2) + cbands]
        loss.append(F.smooth_l1_loss(e_r,rbands.to(device)).view(-1,1))
    
    return torch.mean(torch.cat(loss))

def get_data(raw_dir, save_dir, force_reload=False):
    data = GGCNNDATASET(raw_dir, save_dir, force_reload)
    infos = data.infos
    return data, infos

def evaluate(model, graphs, labels, infos, eikr_m1, eikr_0, eikr_1):
    with torch.no_grad():
        transform, d = read_infos(labels, infos)
        
        transform = transform.to(device)
        d = d.to(device)
        
        output = model(graphs, transform, d)
        loss = compute_loss(output, labels, infos, eikr_m1, eikr_0, eikr_1)
        return loss
    
def read_config(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        config_para = json.load(f)
    return config_para

def main(config_json_file, ribbon_wide):
    
    eikr_m1, eikr_0, eikr_1 = get_eikr()
    eikr_m1 = eikr_m1.to(device)
    eikr_0 = eikr_0.to(device)
    eikr_1 = eikr_1.to(device)
    
    config_para    = read_config(config_json_file)
    
    # configure hyper parameters
    batch_size     = config_para["batch_size"]
    num_epoch      = config_para["num_epoch"]
    lr_radio_init  = config_para["lr_radio_init"]
    num_lr_decay   = config_para["num_lr_decay"]
    lr_radio_decay = config_para["lr_radio_decay"]
    
    # configure trainingset path
    trainset_rawdata_path = f'../data/data set/one-shot/raw/{ribbon_wide}-H-AGNR/'
    trainset_dgldata_path = f'../data/data set/one-shot/dgl/{ribbon_wide}-H-AGNR/'
    
    # configure network structure
    gnn_dim_list           = config_para["gnn_dim_list"]
    gnn_head_list          = config_para["gnn_head_list"]
    onsite_dim_list        = config_para["onsite_dim_list"]
    hopping_dim_list1      = config_para["hopping_dim_list1"]
    hopping_dim_list2      = config_para["hopping_dim_list2"]
    expander_bessel_dim    = config_para["expander_bessel_dim"]
    expander_bessel_cutoff = config_para["expander_bessel_cutoff"]
    
    seed_torch(seed = 24)
    trainset, traininfos = get_data(
                                    raw_dir = trainset_rawdata_path, 
                                    save_dir = trainset_dgldata_path, 
                                    force_reload = True,
                                    )
    
    traingraphs, trainlabels = trainset.get_all()
    traingraphs = dgl.batch(traingraphs)
    traingraphs = traingraphs.to(device)
    train_dataloader = GraphDataLoader(trainset, batch_size = batch_size, drop_last = False, shuffle = False)
    
    model = WHOLEMODEL(
                        gnn_dim_list = gnn_dim_list,
                        gnn_head_list = gnn_head_list,
                        onsite_dim_list = onsite_dim_list,
                        hopping_dim_list1 = hopping_dim_list1,
                        hopping_dim_list2 = hopping_dim_list2,
                        expander_bessel_dim = expander_bessel_dim,
                        expander_bessel_cutoff = expander_bessel_cutoff,
                       )
    
    model = model.to(device)
    
    opt = torch.optim.Adam(model.parameters(),lr_radio_init)
    
    for epoch in range(1,num_epoch + 1):
        
        if epoch % num_lr_decay == 0:
            for p in opt.param_groups:
                p['lr'] *= lr_radio_decay
            
        for graphs, labels in train_dataloader:
            
            transform, d = read_infos(labels,traininfos)
            
            graphs = graphs.to(device)
            transform = transform.to(device)
            d = d.to(device)
        
            output = model(graphs, transform, d)
            loss = compute_loss(output, labels, traininfos, eikr_m1, eikr_0, eikr_1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        
        trainset_loss = evaluate(model, traingraphs, trainlabels, traininfos, eikr_m1, eikr_0, eikr_1)
        print("Epoch {:05d} | TrainSet_Loss {:.6f}" . format(epoch, trainset_loss.item()))

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            }, f'./results/one-shot/{ribbon_wide}-H-AGNR.pkl')

if __name__ == '__main__':
    main(config_json_file = "./script_one_shot_config.json", ribbon_wide = 36)

    


