# -*- coding: utf-8 -*-
"""
Created on MARCH 2025

@author: fperotto
"""

##########################################################
# ONNX UTILS
##########################################################

#import numpy as np
import onnx

##########################################################

def get_inputs(onnx_model: onnx.ModelProto):
    #the name of the input (e.g.: 'input', 'X', etc.)
    #nb_inputs = len(onnx_model.graph.input)
    onnx_input_nodes = []
    for input_node in onnx_model.graph.input:
        onnx_input_nodes.append({"name":input_node.name, "dims": [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]})
    #print("inputs:", onnx_input_nodes)
    return onnx_input_nodes
    
def get_outputs(onnx_model: onnx.ModelProto):
    #the name of the output (e.g.: 'ouput', 'Y', etc.)
    nb_outputs = len(onnx_model.graph.output)
    onnx_output_nodes = []
    for output_node in onnx_model.graph.output:
        onnx_output_nodes.append({"name":output_node.name, "dims": [dim.dim_value for dim in output_node.type.tensor_type.shape.dim]})
    #print("outputs:", onnx_output_nodes)
    return onnx_output_nodes
    
def get_input_dims(onnx_model: onnx.ModelProto):
    return [[dim.dim_value for dim in node.type.tensor_type.shape.dim] for node in onnx_model.graph.input]

def get_output_dims(onnx_model: onnx.ModelProto):
    return [[dim.dim_value for dim in node.type.tensor_type.shape.dim] for node in onnx_model.graph.output]

def get_input_names(onnx_model: onnx.ModelProto):
    return [node.name for node in onnx_model.graph.input]

def get_output_names(onnx_model: onnx.ModelProto):
    return [node.name for node in onnx_model.graph.output]
