
💡 **Notes**
- This is a list accompanying our preprint: https://www.biorxiv.org/content/10.1101/2022.08.31.505981v1 . We focus on deep learning methods for protein design from the last four years. This table complements Table 1 in our manuscript.
- We curated this list manually and as such it might be incomplete. Please drop us an email or open an issue if you find your method missing.
- We order the methods by release date (preprint when available) and categorize them in four classes (for more details on these categories see our [preprint]()):.
  * 1: 'Fixed-backbone' protein design; p(sequence|structure)
  * 2: Structure generation; p(structure)
  * 3: Sequence generation; p(sequence) or p(sequence|sequence*)
  * 4: Concomitant protein and sequence design. p(sequence and structure) (which can be constrained). 
- Others before us have also done a fantastic work assembling other methods, sometimes overlapping with this list. We link these lists here:
  * Kevin Yang's list on [ML methods for protein research](https://github.com/yangkky/Machine-learning-for-proteins/blob/master/README.md)
  * Christian Dallago & Sergey Ovchinnikov's lists on [structure prediction methods](https://github.com/sacdallago/folding_tools) and [protein language models](https://github.com/sacdallago/folding_tools/blob/main/pLM.md).
  * Simon Dürr and Gina El Nesr's list on [inverse folding](https://github.com/duerrsimon/folding_tools/blob/main/inversefolding.md)

**Contributors**
- [@noeliaferruz](https://twitter.com/noeliaferruz)
- [@sacdallago](https://twitter.com/sacdallago)
- [@michaelheinzinger](https://twitter.com/michaelheinzinger)

# Class I: Protein sequence design ("Fixed-backbone")
Methods in this class attempt to solve the classical protein design problem: Find an optimal sequence that adopts a pre-determined 3D structure.
| Name      | Architecture | Number of Parameters | User Input | Output | Training Dataset | Paper | Code | Release Month/Year |
| :-------- | ------------- |---------------------| ---------- | ------- | --------------- | ----- | ---- | ------------------ | 
| **SPIN2** |  FNN | ~105k | 3D structure | sequence | 1,532 X-ray structures | [Paper](https://onlinelibrary.wiley.com/doi/10.1002/prot.25489) | [Code used to be here - no longer available](http://sparks-lab.org/service/) | 2018/02 | 
| **SPROF** |  CNN-LSTM | - | 3D structure | sequence | 1,532 X-ray structures | [Paper](https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00438) | [Code](https://github.com/biomed-AI/SPROF)| 2019/08 | 
| ProDCoNN | CNN | >28k |3D structure | sequence | Several datasets; Largest: 21,071 protein PDB structures | [Paper](https://onlinelibrary.wiley.com/doi/10.1002/prot.25868)|[Web Server](https://prodconn.stat.fsu.edu/)[Code reimplementation](https://github.com/wells-wood-research/timed-design)  |2019/12|
| Ingraham et al. | modified Transformer | >3k | 3D structure | sequence | CATH 4.2 40% sequences/structures |[Paper](https://www.mit.edu/~vgarg/GenerativeModelsForProteinDesign.pdf)| [Code](https://www.mit.edu/~vgarg/github.com/jingraham/neurips19-graph-protein-design) |2019/12|
| Anand et al. | CNN | - | 3D structure | sequence | 53,414  CATH domain structures  |[Paper](https://www.nature.com/articles/s41467-022-28313-9) | [Code](https://github.com/ProteinDesignLab/protein_seq_des)  |2020/01 |
| DenseCPD | CNN | 3M | 3D structure | sequence | 11,227 3D structures  | [Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00043)| [Web server](http://protein.org.cn/densecpd.html) [Code reimplementation](https://github.com/wells-wood-research/timed-design) |2020/01|
| ProteinSolver |   GNN | - |3D structure | sequence | 72,464,122 sequences/adjacency matrices pairs | [Paper](https://www.sciencedirect.com/science/article/pii/S2405471220303276?via%3Dihub)| [Code](https://github.com/ostrokach/proteinsolver) |2020/03|
| GVP | GVP | - | 3D structure | sequence | CATH 4.2 40% sequences/structures | [Paper](https://arxiv.org/pdf/2009.01411)| [Code](https://github.com/drorlab/gvp) |2020/07|
| Norn et al. | CNN | N/A |Distance map? | sequence | N/A | [Paper](https://www.pnas.org/doi/10.1073/pnas.2017228118)| [Code](https://github.com/gjoni/trDesign) |2020/07|
| Fold2Seq | modified Transformer  | - | 3D structure |  sequence | 45,995 3D structures from CATH 4.2 | [Paper](https://arxiv.org/abs/2106.13058)| [Code](https://github.com/IBM/fold2seq) |2021/06|
| CNN_protein_landscape | CNN  | >10M | 3D structure |  sequence | 16,569 PDB chains | [Paper](https://link.springer.com/article/10.1007/s10867-021-09593-6)| [Code](https://github.com/akulikova64/CNN_protein_landscape) |2021/08|
| Orellana et al. | GCN | - | 3D structures | sequence | CATH 4.2 40% sequences/structures | [Paper](https://www.biorxiv.org/content/10.1101/2021.09.06.459171v3)| - |2021/11|
| McPartlon et al. | modified Transformer | >10k | 3D structures | sequences | 37k 3D structures from BC40 | [Paper](https://www.biorxiv.org/content/10.1101/2022.04.15.488492v1)| - |2022/04|
| ESM-IF1  | GVP-Transformer | 142M | 3D structure | sequence | 16k 3D structures + 1.2 M AF2 predictions | [Paper](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v1.full.pdf)| [Code](https://github.com/facebookresearch/esm) | 2022/04 |
| ABACUS-R | Transformer | 152M | 3D structures | sequence | CATH 4.2 | [Paper](ttps://www.nature.com/articles/s43588-022-00273-6)| [Code](https://codeocean.com/capsule/6949436/tree/v1)  |2022/02|
| TERMinator | GNN | - | 3D structure | Potts model | CATH 4.2 40% sequences/structures | [Paper](https://arxiv.org/abs/2204.13048)| |2022/04|
| MIF | SGNN | - | 3D structure | sequence | - | [Paper](https://www.biorxiv.org/content/10.1101/2022.05.25.493516v1)| [Code](https://github.com/microsoft/protein-sequence-models) |2022/05|
| ProteinMPNN | MPNN | >28k |Backbone | sequence | CATH 4.2 40% | [Paper](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)| [Code](https://github.com/dauparas/ProteinMPNN) [Web Interface](https://github.com/dauparas/ProteinMPNN) |2022/07|
| ProDESIGN-LE | - | - | - | sequence | - | [Paper](https://www.biorxiv.org/content/10.1101/2022.06.25.497605v4)| - |2022/07|
| TIMED| CNN | 3M | 3D structures | sequence | 32k sequences from the PISCES dataset (cullpdb_pc90_res3.0_R1.0_d200702_chains40583) | [Paper](https://doi.org/10.5281/zenodo.6997495)| [Code](https://github.com/wells-wood-research/timed-design) |2022/08|



#  Class II: Structure generation
Methods in this class generate structures unconditionally or from a set of secondary structural conditions.

| Name      | Architecture | Number of Parameters | User Input | Output | Training Dataset | Paper | Code | Release Month/Year |
| :-------- | ------------- |---------------------| ---------- | ------- | --------------- | ----- | ---- | ------------------ | 
| **64GAN** |  GAN | - | - | contact map (3D structure via ADMM) | 427,659 contact maps | [Paper](https://papers.nips.cc/paper/2018/hash/afa299a4d1d8c52e75dd8a24c3ce534f-Abstract.html) |-|  2018/12| 
| **64GAN-DIS** | GAN | - | - | distance map (3D structure via CNN) | 800,000 distance maps | [Paper](https://openreview.net/forum?id=SJxnVL8YOV) |  | 2019/03| 
| **RamaNet** |  LSTM  | - | - | A sequence of φ and ψ angles | 607 helical structures | [Paper](https://f1000research.com/articles/9-298) | [Code](https://sarisabban.github.io/RamaNet/) | 19/06 | 
| **DECO-VAE** | VAE  | - | - | contact graph (translatable to contact map) | >650,000 contact graphs | [Paper](https://arxiv.org/abs/2004.07119) | - | 2020/04 | 
| **SCUBA** |  NC-NN | ~20k | secondary structure motif | backbone | 12,465  structures | [Paper](https://www.nature.com/articles/s41586-021-04383-5) | [Code](https://zenodo.org/record/4533424#.YwP3UPFBwqs) | | 
| **GENESIS** |  VAE | - | secondary structure motif | contact map | 40,726 backbones with remodeled loops | [Paper](https://openreview.net/pdf?id=DwN81YIXGQP) | - | 2022/04 | 
| **Ig-VAE** | VAE | - | - | protein backbone coordinates | 10,768 individual immunoglobulin domains | [Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010271) | [Code](https://github.com/ProteinDesignLab/IgVAE) | 2022/06| 
| **Lai et al.** | VAE | - | topology | protein backbone coordinates | CATH 4.2 40% sequences/structures | [Paper](https://www.biorxiv.org/content/10.1101/2022.07.09.499440) | Code release soon | 2022/07| 

# Class III: Sequence generation
Methods in this class generate sequences usually from autoregressive language models, and can sometimes be conditioned.

| Name      | Architecture | Number of Parameters | User Input | Output | Training Dataset | Paper | Code | Release Month/Year |
| :-------- | ------------- |---------------------| ---------- | ------- | --------------- | ----- | ---- | ------------------ | 
| **ProteinGAN** | GAN  | 60M |   | sequence |  16,706 MDH sequences | [Paper](https://www.nature.com/articles/s42256-021-00310-5) | [Code](https://github.com/Biomatter-Designs/ProteinGAN) | 2019/03 | 
| **ProGen** | Transformer  | 1.2B | Optional: sequence or function | sequence | 280M sequences | [Paper](https://www.biorxiv.org/content/10.1101/2020.03.07.982272v2) | [Code](https://github.com/salesforce/progen) | 2020/03 | 
| **ProtXLnet** |  Transformer |409M | Optional: sequence | sequence | UniRef100 | [Paper](https://ieeexplore.ieee.org/document/9477085) | [Code](https://huggingface.co/Rostlab/prot_xlnet) | 2020/07 | 
| **EVE** | VAE |  | MSA | Sequence | 3,219 MSAs | [Paper](https://www.nature.com/articles/s41586-021-04043-8) | [Code](https://github.com/OATML-Markslab/EVE) | 2020/12 | 
| **DARK3** | Transformer  | 110M | Optional: sequence | sequence | 615,000 synthetic sequences | [Paper](https://www.biorxiv.org/content/10.1101/2022.01.27.478087v1.full) | - | 2022/01 | 
| **ProtGPT2** | Transformer | 739M | Optional: sequence | sequence | UniRef50 | [Paper](https://www.nature.com/articles/s41467-022-32007-7) | [Code](https://huggingface.co/nferruz/ProtGPT2) | 2022/03 | 
| **RITA** |  Transformer |  1.2B | Optional: sequence | sequence | UniRef50 | [Paper](https://arxiv.org/abs/2205.05789) | [Code](https://huggingface.co/lightonai/RITA_xl) | 2022/05 | 
| **ProGEN2** | Transformer  | 6.4B | Optional: sequence | sequence | Uniref90+BF30 | [Paper](https://arxiv.org/abs/2206.13517) | [Code](https://github.com/salesforce/progen) | 2022/06 |

# Class IV: Sequence and structure design
Methods in this class generate sequences and structures concomitantly, and include hallucination methods and constrained generation (inpainting)

| Name      | Architecture | Number of Parameters | User Input | Output | Training Dataset | Paper | Code | Release Month/Year |
| :-------- | ------------- |---------------------| ---------- | ------- | --------------- | ----- | ---- | ------------------ | 
| **Hallucination** | CNN (trRosetta)  | N/A | random sequence | sequence/structure | N/A | [Paper](https://www.nature.com/articles/s41586-021-04184-w) | [Code](https://github.com/gjoni/trDesign) | 2020/07 | 
| **Constrained hallucination** |  CNN (trRosetta) | N/A | sequence/structure | sequence/structure | N/A | [Paper](https://pubmed.ncbi.nlm.nih.gov/34547592/) | [Code](https://www.biorxiv.org/content/10.1101/2020.11.29.402743v1.full) | 2020/11 | 
| **Constrained hallucination2** |  CNN (RoseTTAFold) | N/A | sequence/structure | sequence/structure | N/A | [Paper](https://www.science.org/doi/full/10.1126/science.abn2100?af=R) | [Code](https://github.com/RosettaCommons/RFDesign) | 2021/11 | 
| **RFjoint** | CNN (RoseTTAFold, finetuned) | N/A | sequence/structure | sequence/structure | Finetuned with 25% PDB version 02/2020 + 75 % AF2 structures  | [Paper](https://www.science.org/doi/full/10.1126/science.abn2100?af=R) | [Code](https://github.com/RosettaCommons/RFDesign) | 2021/11 | 
| **Protein Diffusion** | Diffussion model  | - | Secondary structure motif sketches | sequence/structure | 53,414 3D structures (95% CATH 4.2 S95) | [Paper](https://arxiv.org/abs/2205.15019) | [Code](https://nanand2.github.io/proteins/) | 2022/05| 

