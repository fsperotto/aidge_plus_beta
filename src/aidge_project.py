# -*- coding: utf-8 -*-
"""
Created on 24 June 2024

@author: fperotto
"""

##########################################################
# AIDGE PROJECT
##########################################################

from datetime import datetime
import os
from io import StringIO 
import sys
from copy import deepcopy, copy

import hashlib

import aidge_core
import aidge_onnx

from aidge_utils import str_aidge_graph_structure, list_files, freeze_producers, fix_export

##########################################################

class CaptureStdOut(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

##########################################################

def get_filehash(filepath, buffer_size=65536):
    
    md5 = hashlib.md5()
    #sha1 = hashlib.sha1()
    #sha256 = hashlib.sha256()

    #with open(filepath, 'rb', buffering=0) as f:
    #    return hashlib.file_digest(f, 'sha256').hexdigest()
    
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            md5.update(data)
            #sha1.update(data)
            #sha256.update(data)

    #print("MD5: {0}".format(md5.hexdigest()))
    #print("SHA1: {0}".format(sha1.hexdigest()))
    #print("SHA256: {0}".format(sha256.hexdigest()))
    
    return md5.hexdigest()
    
##########################################################

def get_author(author:str=None):
    if author:
        return author
    else:
        try:
            return os.getlogin()
        except:
            return "unknown"

##########################################################

class AidgeProject():

    ##########################################################

    def __init__(self, name:str="My aidge project", author:str=None):
    
        self.name = name
        self.author = get_author(author)
        self.colaborators = []
        self.creation_date = datetime.now()
        self.operational_domain = None
        self.data = None
        self.model = None
        self.model_development_history = []
        self.retained_target_architecture = None
        self.declined_target_architectures = []
        self.log = [{"message":"project created",
                     "op_log":None,
                     "author":self.author,
                     "timestamp":self.creation_date}]

    ##########################################################

    def load(self, project_folder):
        pass

    def save(self, project_folder):
        pass

    ##########################################################

    def import_onnx(self, onnx_filepath, author:str=None):
    
        with CaptureStdOut() as output:
            print("Opening onnx file:", onnx_filepath)
            #FILE PROPERTIES
            fileinfo = os.stat(onnx_filepath)
            print("file size:", fileinfo.st_size)
            print("file creation time:", datetime.fromtimestamp(fileinfo.st_ctime).strftime('%Y-%m-%d %H:%M:%S'))
            print("file modification time:", datetime.fromtimestamp(fileinfo.st_mtime).strftime('%Y-%m-%d %H:%M:%S'))
            print("file md5 hash:", get_filehash(onnx_filepath))
            valid = aidge_onnx.check_onnx_validity(onnx_filepath)
            #OPEN
            model = aidge_onnx.load_onnx(onnx_filepath)
            coverage = aidge_onnx.has_native_coverage(model)
            (native_nodes_types, generic_node_types) = aidge_onnx.native_coverage_report(model)
            #model.save(output_folder + "loaded_model")
            #print graph structure
            print("--------------------------")
            print("Model:")
            print(str_aidge_graph_structure(model))
            print("[OK]")
        
        if self.model is not None:
            self.model_development_history += [self.model]
            
        self.model = model
        
        self.add_log(message=f"onnx imported as model ({onnx_filepath})", op_log=output, author=author)

    ##########################################################

    def apply_recipe(self, recipe=None, author:str=None):
    
        with CaptureStdOut() as output:
            print("--------------------------")
            print("Applying recipes...")
            aidge_core.remove_flatten(self.model)
            aidge_core.fuse_mul_add(self.model)
            aidge_onnx.native_coverage_report(self.model)
            print("[OK]")

        self.add_log(message="Recipes applied", op_log=output, author=author)

    ##########################################################


    def add_log(self, message:str, author=None, op_log=None, timestamp=None):
        self.log.append({"message":message,
                         "op_log":op_log,
                         "author":get_author(author),
                         "timestamp":datetime.now()})
    
    def get_log(self, op_logs=True):
        log_rows = []
        for i, item in enumerate(self.log):
            row = f"{i} [{item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}]: {item['message']} (author: {item['author']}); "
            if op_logs and item['op_log']:
                row += '\n | ' + '\n | '.join(item['op_log'])
            log_rows.append(row)
        return '\n'.join(log_rows)
        
    def __str__(self):
        return f"""
Name: {self.name};
Author: {self.author};
Colaborators: {self.colaborators};
Creation date and time: {self.creation_date.strftime("%Y-%m-%d %H:%M:%S")};
Model: {self.data};
Data: {self.model};
Operational Domain (ODD): {self.operational_domain};
Retained Target Hardware Architecture: {self.retained_target_architecture};
"""