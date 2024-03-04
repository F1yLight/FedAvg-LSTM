import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy
import argparse
import random
import numpy as np
from time import time
import pandas as pd
from typing import List

def xlsx_read(path, col_names: List[str] = [], row_begin=2, row_end=-1, sheet_name='Sheet1') -> List[List[str]]:
    data = []
    for col_name in col_names:
        d = pd.read_excel(path, sheet_name=sheet_name)[col_name]
        d = [float(x) for x in d]
        data.append(d)
    if row_end == -1: 
        row_end = len(d) + 1
    return list(d[row_begin - 2: row_end - 1])

def process_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
        print(data)


def main():
    process_data('pems_output.xlsx')
    pass

if __name__ == '__main__':
    main()