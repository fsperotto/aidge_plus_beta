# -*- coding: utf-8 -*-
"""
Created on 26 June 2024

@author: fperotto
"""

##########################################################

import os
import numpy as np

from aidge_project import AidgeProject

myproject = AidgeProject("Test Project", author="Myself")
print(myproject)

myproject.import_onnx("./models/acasxu.nnet.onnx")
print(myproject)

myproject.apply_recipe()
print(myproject)

print("log:")
print(myproject.get_log())
print()
