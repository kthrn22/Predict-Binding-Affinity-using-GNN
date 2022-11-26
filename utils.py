import numpy as np
from rdkit.Chem.rdmolfiles import MolFromMol2File, MolFromPDBFile
from rdkit.Chem import SDMolSupplier

def ligand_atom_to_feature(atom):
    #return [atom.get_atom_type(),
    #        atom.get_degree(),
    #        atom.get_Hydrogen_attached(),
    #        atom.is_aromatic(),]

    return [#atom_types[1] if atom.GetAtomicNum() not in atom_types else atom_types[atom.GetAtomicNum()],
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(includeNeighbors = True),
            atom.GetImplicitValence(),
            int(atom.GetIsAromatic()),]

def ligand_to_graph(ligand):
    # atom
    atoms_feature_list = [ligand_atom_to_feature(atom) for atom in ligand.GetAtoms()]
    node_features = np.array(atoms_feature_list, dtype = np.float64)

    c = ligand.GetConformer()
    coordinates = [[c.GetAtomPosition(atom_index)[i] for i in range(3)] for atom_index in range(ligand.GetNumAtoms())]
    node_positions = np.array(coordinates, dtype = np.float64)

    # bond
    if ligand.GetNumBonds() > 0:
        edge_list = []

        for bond in ligand.GetBonds():
            atom_u, atom_v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list.append((atom_u, atom_v))
            edge_list.append((atom_v, atom_u))

        edge_index = np.array(edge_list, dtype = np.int64).T

    else:
        edge_index = np.empty((2, 0), dtype = np.int64)

    graph = {"node_feat": node_features,
            "num_nodes": node_features.shape[0],
            "edge_index": edge_index,
            "node_positions": node_positions}

    return graph

def protein_atom_to_feature(atom):
    atom_types = {6: 0, 7: 1, 8: 2, 16: 3, 9: 4, 15: 5, 17: 6, 35: 7, 5: 8, 1: 9}

    return [#atom_types[1] if atom.GetAtomicNum() not in atom_types else atom_types[atom.GetAtomicNum()],
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(includeNeighbors = True),
            atom.GetImplicitValence(),
            int(atom.GetIsAromatic()),]

def protein_to_graph(protein):
    # atom
    atoms_feature_list = [protein_atom_to_feature(atom) for atom in protein.GetAtoms()]
    node_features = np.array(atoms_feature_list, dtype = np.float64)

    c = protein.GetConformer()
    coordinates = [[c.GetAtomPosition(atom_index)[i] for i in range(3)] for atom_index in range(protein.GetNumAtoms())]
    node_positions = np.array(coordinates, dtype = np.float64)

    # bond
    if protein.GetNumBonds() > 0:
        edge_list = []

        for bond in protein.GetBonds():
            atom_u, atom_v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list.append((atom_u, atom_v))
            edge_list.append((atom_v, atom_u))

        edge_index = np.array(edge_list, dtype = np.int64).T

    else:
        edge_index = np.empty((2, 0), dtype = np.int64)

    graph = {"node_feat": node_features,
            "num_nodes": node_features.shape[0],
            "edge_index": edge_index,
            "node_positions": node_positions}

    return graph

def covalent_interactions_graph(ligand_graph, protein_graph):
    node_features = np.concatenate([ligand_graph["node_feat"], protein_graph["node_feat"]], axis = 0)
    num_nodes = ligand_graph["num_nodes"] + protein_graph["num_nodes"]
    edge_index = np.concatenate([ligand_graph["edge_index"], protein_graph["edge_index"] + ligand_graph["num_nodes"]], axis = 1)
    node_positions = np.concatenate([ligand_graph["node_positions"], protein_graph["node_positions"]], axis = 0)

    graph = {"node_feat": node_features,
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "node_positions": node_positions}

    return graph

def covalent_and_intermolecular_interactions_graph(ligand_graph, protein_graph):
    node_features = np.concatenate([ligand_graph["node_feat"], protein_graph["node_feat"]], axis = 0)
    num_nodes = ligand_graph["num_nodes"] + protein_graph["num_nodes"]
    
    intermolecular_bonds, weight = [], []
    for ligand_atom_id in range(ligand_graph["num_nodes"]):
        for protein_atom_id in range(protein_graph["num_nodes"]):
            ligand_atom_pos = ligand_graph["node_positions"][ligand_atom_id]
            protein_atom_pos = protein_graph["node_positions"][protein_atom_id]
            distance = sum((ligand_atom_pos - protein_atom_pos) ** 2)
            if distance < 5:
                intermolecular_bonds.append((ligand_atom_id, protein_atom_id + ligand_graph["num_nodes"]))
                intermolecular_bonds.append((protein_atom_id + ligand_graph["num_nodes"], ligand_atom_id))
                weight.append(distance)
                weight.append(distance)

    if len(intermolecular_bonds) > 0:
        intermolecular_edge_index = np.array(intermolecular_bonds, dtype = np.int64).T
        intermolecular_edge_weight = np.array(weight, dtype = np.float64).reshape(1, len(intermolecular_bonds))
    else:
        intermolecular_edge_index = np.empty((2, 0), dtype = np.int64)
        intermolecular_edge_weight = np.empty((1, 0), dtype = np.float64)

    num_covalent_bonds = ligand_graph["edge_index"].shape[1] + protein_graph["edge_index"].shape[1]
    edge_index = np.concatenate([ligand_graph["edge_index"], protein_graph["edge_index"] + ligand_graph["num_nodes"],
                                intermolecular_edge_index], axis = 1)
    edge_weight = np.concatenate([np.array([[1] * ligand_graph["edge_index"].shape[1]]), np.array([[1] * protein_graph["edge_index"].shape[1]]),
                                intermolecular_edge_weight], axis = 1).reshape(edge_index.shape[1])
    node_positions = np.concatenate([ligand_graph["node_positions"], protein_graph["node_positions"]], axis = 0)

    graph = {
        "node_feat": node_features,
        "num_nodes": num_nodes,
        "num_covalent_bonds": num_covalent_bonds,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "node_positions": node_positions,
    }

    return graph

def get_binding_affinity(filename):
    '''
    Get the filename, output the list of kd/ki value corresponds to the complex file
    The affinity is in log(Ki) of log(Kd)
    '''
    map_complex_to_affinity = {}
    with open(filename) as f:
        for line in f:
            if line.startswith("#"):
                continue
        
            line = line.split()
            pdb_id, binding_affinity = line[0], line[3]

            map_complex_to_affinity[pdb_id] = float(binding_affinity)
        
    return map_complex_to_affinity