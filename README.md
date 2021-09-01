# Machine learning framework for electronic band structure prediction

This repo shows the PyTorch implementation of ML framework for predicting band structure, taking the example of graphene nanoribbon systems.  
As this work is associtaed with a manuscript for academic publication, we therefore also inculde all the related files that are necessary to reproduce the results in the manuscript. 
The manuscript doi will be shown here after publication.

The code directory contains the code implementation of our ML framewrok, and the data directory contains all the related data files.

## Description
Our ML framework is empowered by two core building blocks, the graph convolutional network and the LCAO Hamiltonian construction process. 
The former can handle the graph representation of materials with varying sizes and compositions, while the latter enable computing eigenenergies as desired k points.
They together endow our ML framework with the ability to build a physically meaningful mapping from geometric structures to electronic structures, where both the inputs and outputs are highly variable and hard to encoded in to fixed-length representation.

## Requirements

dgl 0.6.0

pytroch 1.8.0

pymatgen 2022.0.3

## Reproduction

All raw data used in our study are provided in data folder, including the input and output files of QE, Wannier90, and NanoTCAD ViDES.
The network structure, hyper parameter setting, and trained network parameters are given in code folder. 

You may run "predict.py" to see the predicted band structures with comparison to ab initio ones. 
Change the'predict_structure_cif_path' and 'references' for predicting for other graphene nanoribbon systems other than the given example. 



