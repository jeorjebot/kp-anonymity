# kp-anonymity

## Dataset
- https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly#

## Usage 
```
[*] usage: python kp-anonymity.py k_value p_value paa_value Dataset\Sales_Transaction_Dataset_Weekly_Final.csv
```

### Explain Parameters
- k_value (value of k-anonymity)
- p_value (value of p-anonymity, pattern)
- paa_value (To reduce the dimensionality of patterns)

## Requirements
```
pip install -r requirements.txt
```

- numpy==1.16.4
- pandas==0.25.0
- loguru==0.3.2
- saxpy==1.0.1.dev167 (https://github.com/seninp/saxpy)