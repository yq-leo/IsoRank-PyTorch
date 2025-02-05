# IsoRank-Python
PyTorch Implementation of IsoRank in "[Global alignment of multiple protein interaction networks with application to functional orthology detection](https://doi.org/10.1073/pnas.0806627105)" in PNAS 2008

## Requirements
- numpy
- torch

## Available Datasets
- phone-email
- foursquare-twitter
- ACM-DBLP

## How to use
Run the following command to run IsoRank, replace `{dataset}` with your dataset name:
```sh
python main.py --dataset={dataset}
```
If using gpu, add `--gpu` to the command:
```sh
python main.py --dataset={dataset} --gpu
```