# IsoRank-Python
PyTorch Implementation of IsoRank in "[Global alignment of multiple protein interaction networks with application to functional orthology detection](https://doi.org/10.1073/pnas.0806627105)" in PNAS 2008. The official website is [here](https://cb.csail.mit.edu/mna/)

## Requirements
- numpy
- torch

## Available Datasets
- phone-email
- foursquare-twitter
- ACM-DBLP

## How to use
To run IsoRank, execute the following command and replace `{dataset}` with your dataset name:
```sh
python main.py --dataset={dataset}
```
If using gpu, add `--gpu` to the command:
```sh
python main.py --dataset={dataset} --gpu
```
