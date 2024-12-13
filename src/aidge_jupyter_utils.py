# -*- coding: utf-8 -*-
"""
Created on 4 July 2024

@author: fperotto
"""

##########################################################
# AIDGE JUPYTER UTILS
##########################################################

import base64
from IPython.display import Image, display
#import matplotlib.pyplot as plt

##########################################################

#AUXILIARY PLOT MERMAID GRAPHS

def plot_mmd(path_to_mmd):
  with open(path_to_mmd, "r") as file_mmd:
    graph_mmd = file_mmd.read()
    plot_mmd_str(graph_mmd)

def plot_mmd_str(mmd_str):
  #display(Image(url="https://mermaid.ink/img/"+base64.b64encode(mmd_str.encode("ascii")).decode("ascii")))
  display(Image(url="https://mermaid.ink/img/"+base64.b64encode(mmd_str.encode("utf8")).decode("utf8")))