# GLADPro is a prototype-based method for graph-level anomaly detection with global-level explanations.
 
This is the code for GLADPro

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

python3 main.py --n_prot 3 --regular 500 --hidden_dim 128 --out_dim 64

```
For MUTAG dataset, run with 
```

python3 main.py --epochs 500 --lr 1e-4

```
For PROTEIN dataset, run with 
```

python3 main.py --hidden_dim 128 --out_dim 64 --epochs 500 

```
For DD dataset, run with 
```

python3 main.py --regular 100 --epochs 500 

```
For IMDB-BINARY dataset, run with 
```

python3 main.py --regular 100 --epochs 100 

```