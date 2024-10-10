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

For visulization top-k (i.e., global-level explanations) on mutagen, run
```
python3 visulization
```

All hyper-parameter analysis results are avaliable on parameter_analysis/ fold
contamination results are avaliable on contamination/ fold