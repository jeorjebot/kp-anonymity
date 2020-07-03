# kp-anonymity

## Datasets
- [Sales transactions over a year - UCI](https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly#)
- [News social feedback over eight months - UCI](https://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms)

## How to launch the tool 
```
[*] usage: python3 kp-anonymity.py k_value p_value paa_value dataset_path dataset_output_path
```
## Example
```
python3 kp-anonymity.py 10 2 5 Dataset/Input/Sales_Transaction_Dataset_Weekly_Final.csv Dataset/Anonymized/output.csv
```
### Explain Parameters
- k_value (value of k-anonymity)
- p_value (value of p-anonymity, pattern)
- paa_value (To reduce the dimensionality of patterns) [more on this](https://vigne.sh/posts/piecewise-aggregate-approx/)

## Requirements
```
pip install -r requirements.txt
```
- numpy==1.18.1
- pandas==1.0.3
- loguru==0.5.1
- saxpy==1.0.1.dev167 (https://github.com/seninp/saxpy)
- pathlib==1.0.1
- matplotlib==3.2.2