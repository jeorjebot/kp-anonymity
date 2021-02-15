# kp-anonymity
 take a look at the [presentation!](./Presentation/kp-anonym-presentation.pdf)
 
## Paper
This work is based on the novel approach for the anonymization of time series with a special focus on the pattern loss, [presented by this paper](https://ieeexplore.ieee.org/document/6095556)

## Datasets
- [Sales transactions over a year - UCI](https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly#)
- [News social feedback over eight months - UCI](https://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms)

## Requirements
```console
foo@bar:~$ python3 -m pip install -r requirements.txt
```
- numpy==1.18.1
- pandas==1.0.3
- loguru==0.5.1
- saxpy==1.0.1.dev167 (https://github.com/seninp/saxpy)
- pathlib==1.0.1
- matplotlib==3.2.2

## How to launch the tool 
```console
foo@bar:~$ python3 kp-anonymity.py algorithm k_value p_value paa_value dataset_path dataset_output_path
```
### Explain Parameters
- **kp-anonimity** : main program
- **algorithm** : choose between *naive* or *kapra* approach
- **k_value** : value of k-anonymity
- **p_value** : value of p-anonymity, pattern
- **paa_value** : to reduce the dimensionality of patterns ([see how](https://vigne.sh/posts/piecewise-aggregate-approx/))
- **dataset_path** : csv input file
- **dataset_output_path** : csv output file

## Example
```console
foo@bar:~$ python3 kp-anonymity.py kapra 10 2 5 Dataset/Input/Sales_Transaction_Dataset_Weekly_Final.csv Dataset/Anonymized/output.csv
```
## Time Utility
To compare time scalability between **naive** and **kapra** approaches, launch the *test* utility:
```console
foo@bar:~$ cd Utility
foo@bar:~$ ./test.sh
```
## Repository Structure
- **Dataset**: contains the datasets used in my tests
    - **Input**: input datasets for the tool
    - **Anonymized**: store the output of the tool
- **Paper**: contains the two papers studied for this project
    - [Utility-Based Anonymization for Privacy Preservation with Less Information Loss](https://www.cs.sfu.ca/~jpei/publications/localrecoding-sigkddExp06.pdf)
    -  [Supporting Pattern-Preserving Anonymization for Time-Series Data](https://ieeexplore.ieee.org/document/6095556)
- **Utility**: contains scripts for verify time efficiency and for resetting the tool
- **kp-anonymity.py**: main script, which implement *naive* and *kapra* algorithms
- **node.py**: manage the *create-tree phase* of both algorithms
- **dataset_anonymized.py**: manage the printing and anonymized value replacing of the output dataset
- **create_dataset.py**: script for generating subdataset of the News Social Dataset, used for measure time scalability of both algorithms  
- **requirements.txt**: list of necessary packages

## Author and Credits
**Author: Giorgio Rossi, student of Computer Engineering (LM) - UNIGE - a.y. 2019/2020**.

Final project of the course *Data Protection and Privacy*.
Work based on [Davide Caputo's repository](https://github.com/Dado1513/kp-anonymity-to-complete).
