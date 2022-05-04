""" A dataset parser that reads images from folders

Folders are scannerd recursively to find image files. Labels are based
on the folder hierarchy, just leaf folders by default.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os


from .parser import Parser
import random
import numpy as np








def read_list(path):
    imgs = []
    with open(path, 'r') as f:
        for line in f:
            name, target = line.split(' ')
            imgs.append((name, int(target)))
    return imgs


class ParserTxtFile(Parser):

    def __init__(
            self,
            root,
            train_val_list,
            class_map='',
            ):
        super().__init__()

        self.root = root
        self.samples = []
        if  isinstance(train_val_list, str):
            self.samples = read_list(train_val_list)
        elif isinstance(train_val_list, list):
            self.samples = train_val_list
        else:
            raise ValueError('the input must be the path of your train/val list')
        #random.shuffle(self.samples)


    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
