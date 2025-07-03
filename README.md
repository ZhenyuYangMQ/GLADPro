# Global Interpretable Graph-level Anomaly Detection Via Prototype
 
This is the code for GLADPro - KDD-2025. Thanks to the [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) library.

The paper is now available at [ACM](). If you use our code or results, please kindly cite our paper.

```
@inproceedings{yang2025gladpro,
  title={Global Interpretable Graph-level Anomaly Detection via Prototype},
  author={Yang, Zhenyu and Zhang, Ge and Wu, Jia and Yang, Jian and Xue, Shan and Beheshti, Amin and Peng, Hao and Sheng, Quan Z.},
  booktitle={Proc. SIGKDD},
  pages={1--12},
  year={2025}
}
```

## System requirement

#### Programming language
```
Python 3.10.0
```
#### Python Packages
```
PyTorch 1.12.0
CUDA 11.6.1
torch-geometric 2.2.0
numpy   1.22.3
scikit-learn 1.1.1
scipy   1.8.0
networkx  2.8.4
```

## Run the demo code

For Mutagen datasets, run with defualt setting
```
python3 main.py
```
For BA-TYPE dataset, run with 
```
python3 main.py --dataset BA-TYPE --n_prot 3 --regular 500 --hidden_dim 128 --out_dim 64
```
For MUTAG dataset, run with 
```
python3 main.py --dataset MUTAG --epochs 500 --lr 1e-4
```
For PROTEIN dataset, run with 
```
python3 main.py --dataset PROTEINS --hidden_dim 128 --out_dim 64 --epochs 500 
```
For DD dataset, run with 
```
python3 main.py --dataset DD --regular 100 --epochs 500 
```
For IMDB-BINARY dataset, run with 
```
python3 main.py --dataset IMDB-BINARY --regular 100 --epochs 100 
```