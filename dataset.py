import rdkit
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit.Chem.rdmolfiles import MolFromMol2File, MolFromPDBFile
from rdkit.Chem import SDMolSupplier

import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from utils import *


class Ligand_Protein_Dataset(InMemoryDataset):
    def __init__(self, root, data_dir, affinity_file, transform = None, pre_transform = None):

        self.root = root
        self.affinity_file = affinity_file
        self.data_dir = data_dir
        
        super(Ligand_Protein_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ds_raw"

    @property
    def processed_file_names(self):
        return "ds.pt"

    def download(self):
        pass

    def process(self):
        data_list = []

        map_complex_to_affinity = get_binding_affinity(self.affinity_file)

        for i in tqdm(range(len(os.listdir(self.data_dir)))):
            complex_folder_name = os.listdir(self.data_dir)[i]
            if complex_folder_name.startswith("."):
                continue

            pdb_id = complex_folder_name
            
            # ligand structure in mol2 file
            #ligand_structure_file_path = os.path.join(self.root, complex_folder_name, "{}_ligand.mol2".format(pdb_id))
            #ligand_info = process_ligand_mol2(ligand_structure_file_path)
            #ligand = Ligand(ligand_info[0], ligand_info[1], ligand_info[2])

            #ligand = MolFromMol2File(ligand_structure_file_path, sanitize = False)
            #if ligand is None:
            #    continue
            #ligand_graph = ligand_to_graph(ligand)

            ligand_structure_file_path = os.path.join(self.data_dir, complex_folder_name, "{}_ligand.sdf".format(pdb_id))
            ligand = next(SDMolSupplier(ligand_structure_file_path, sanitize = False))
            if ligand is None:
                continue
            ligand_graph = ligand_to_graph(ligand)

            # protein structure in pdb file
            protein_structure_file_path = os.path.join(self.data_dir, complex_folder_name, "{}_protein.pdb".format(pdb_id))
            protein = MolFromPDBFile(protein_structure_file_path, sanitize = False)
            if protein is None:
                continue
            protein_graph = protein_to_graph(protein)

            graph = covalent_and_intermolecular_interactions_graph(ligand_graph, protein_graph)
            binding_affinity = map_complex_to_affinity[pdb_id]

            assert(graph["node_feat"].shape[0] == graph["num_nodes"])
            #assert(graph["edge_index"].shape[1] == graph["edge_weight"].shape[1])
            #print("Done assert")

            data = Data()

            data.__num_nodes__ = graph["num_nodes"]       
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.float32)
            edge_index = torch.from_numpy(graph["edge_index"]).to(torch.long)
            data["edge_index_1"] = edge_index[:, : graph["num_covalent_bonds"]]
            data["edge_index_2"] = edge_index
            data["edge_weight"] = torch.from_numpy(graph["edge_weight"]).to(torch.float32)
            data.y = torch.Tensor([binding_affinity]).to(torch.float32)
            #data["num_covalent_bonds"] = graph["num_covalent_bonds"]
            #data.pos = torch.from_numpy(graph["node_positions"]).to(torch.float32)
            
            data_list.append(data)

        data, slices = self.collate(data_list)

        print("Done!")

        torch.save((data, slices), self.processed_paths[0]) 