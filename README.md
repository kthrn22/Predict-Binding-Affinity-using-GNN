# Predict-Binding-Affinity-using-GNN
Binding affinity is the strength of the binding interaction between a biomolecule (e.g. protein or RNA) and its ligand partner (e.g. drug or inhibitor). Using binding affinity as measurement can help design drugs that bind selectively to their target. This project aims to predict the binding affinity of protein-ligand complexes using Graph Neural Networks, and the model is implemented based on the method in [Predicting drug-target interaction using 3D structure-embedded graph representations from graph neural networks](https://arxiv.org/abs/1904.08144)

## Model's Architecture

* Embed the 3D structure of a protein-ligand complex

$\mathbf{A}^1$: adjacency matrix for covalent bond (only consider intramolecular forces)

$$
\begin{split}
    \mathbf{A}^1_{ij} = 
        \begin{cases}
            1 & \text{if i and j are connected by covalent bond or i = j} \\
            0 & \text{otherwise} \\ 
        \end{cases}
\end{split}
$$

$\mathbf{A}^2$: adjacency matrix for covalent and non-covalent bond (consider intermolecular + intramolecular forces)

$$
\begin{split}
    \mathbf{A}^2_{ij} = 
        \begin{cases}
            \mathbf{A}^1_{ij} & \text{if i and j are both protein atoms or both ligand atoms} \\
            e^{-(d_{ij} - \alpha)^2 / \beta} & \text{if } d_{ij} < 5 \text{ and i and j do not lie in the same molecule} \\   
            0 & \text{otherwise}
        \end{cases}
\end{split}
$$

where $\alpha, \beta$ are learnable parameters, and $e^{-(d_{ij} - \alpha)^2 / \beta}$ represents the strength of intermoleculars bonds: indicates that their strength is weaker than covalent bonds, and as the distance increases, the strength gets weaker

* Graph-attention Layer

Input: node features $\mathbf{X_{\text{in}}} = \{\mathbf{x_1}, \dots, \mathbf{x_N}\}$ with $\mathbf{x_i} \in \mathbb{R}^F$ ($F$ is the number of features, $N$ is the number of nodes)

Transform each node by a learable weight matrix $W \in \mathbb{R}^{F \times F}$: 
$$\mathbf{x_i} = W\mathbf{x_i}$$

Compute attention coefficient (the importand of $i^{th}$ node feature to $j^{th}$ node feature): 
$$e_{ij} = e_{ji} = \mathbf{x}^{T}_i \mathbf{E} \mathbf{x}_j + \mathbf{x}^{T}_j \mathbf{E} \mathbf{x}_i$$

with $\mathbf{E} \in \mathbb{R}^{F \times F}$ is a learnable matrix, only compute $e_{ij}$ if $\mathbf{A_{ij}} = \mathbf{A_{ji}} >0 $

Normalize attention coefficient: 
$$a_{ij} = \frac{\exp(e_{ij})}{\sum_{j \in N(i)} \exp(e_{ij})} \mathbf{A_{ij}}$$

Update: 

$$\hat{\mathbf{x_i}} = \sum_{j \in N(i)} a_{ij} \mathbf{x_j}$$ 

Gated: 

$$z_i = \sigma((\text{CONCAT}[\mathbf{x_{in}}, \mathbf{\hat{x_i}}]) \mathbf{U} + b)$$

where $\mathbf{U} \in \mathbb{R}^{2F \times 1}$ is a learnable matrix, $b$ is a learnable scalar value, $\sigma$ is sigmoid function

Finalize: 

$$ \mathbf{x_{out}} = z_i \mathbf{x_{in}} + (1- z_i) \mathbf{\hat{x_i}} $$

where $z_{i}$ controls how much should the input $(\mathbf{x_{in}})$ should be delivered directly to the next layer

* Architecture:

$\mathbf{x_{out}^1} = Graph-attention-layer(\mathbf{A^1}, \mathbf{x})$, $\mathbf{x_{out}^2} = Graph-attention-layer(\mathbf{A^2}, \mathbf{x})$

$\mathbf{x_{out}} = \mathbf{x_{out}^2} - \mathbf{x_{out}^1}$

Subtracting 2 node features, the model will learn the differences when the protein and ligand binds $(\mathbf{x_{out}^2})$ and when they are seperated $(\mathbf{x_{out}^1})$

Representation of the ligand-protein complex:
$\mathbf{x_{complex}} = \sum \mathbf{x_{out}}$ 

The aboved representation can be fed into an MLP to predict the binding affinity (as a regression task)

## Data processing

All data processing functions in ```utils.py``` is specialized for PDBbind Dataset. After PDBbind Dataset is downloaded, ```root```, ```data_dir```, and ```affinity_file``` in ```config.py``` should be changed based on the dataset's location.

## Parameters and Model Training

All of the parameters can be changed by modifying the ```config.py``` file, and the model can be trained by running ```main.py```

## References

Jaechang Lim, Seongok Ryu, Kyubyong Park, Yo Joong Choe, Jiyeon Ham, and Woo Youn Kim. 2019. Predicting drug–target interaction using a novel graph neural network with 3D structure-embedded graph representation. Journal of chemical information and modeling 59, 9 (2019), 3981–3988. https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00387
