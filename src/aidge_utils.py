# -*- coding: utf-8 -*-
"""
Created on 24 June 2024

@author: fperotto
"""

##########################################################
# AIDGE UTILS
##########################################################

import os
import json

##########################################################

#PRINT AIDGE GRAPH TO TEXT (from node)
def str_aidge_node_parents(node, index=0, nb_brothers=0, str_indent=''):
    if node is not None:
        s = '├' if index<nb_brothers else '└'
        str_result = f'{str_indent}{s} {node.name()} ({node.type()})\n'
        parents = node.get_parents()
        nb_parents = len(parents)
        s = '|' if index<nb_brothers else ' '
        for i, parent in enumerate(parents):
            str_result += str_aidge_node_parents(parent, i, nb_parents-1, f'{str_indent}{s} ')
        return str_result
    else:
        s = '├' if index<nb_brothers else '└'
        return f'{str_indent}{s} <input>\n'


#PRINT AIDGE GRAPH TO TEXT (from model outputs)
def str_aidge_graph_structure(model):
    output_nodes = model.get_output_nodes()
    nb_output_nodes = len(output_nodes)
    str_result = ""
    for i, node in enumerate(output_nodes):
        str_result += str_aidge_node_parents(node, i, nb_output_nodes-1)
    return str_result

##################################################################################################

#PRINT AIDGE STATIC SCHEDULING TO TEXT (from scheduler)
def str_aidge_seq_scheduling(static_scheduling):
    str_result = ""
    for i, node in enumerate(static_scheduling):
        str_result += '  '*i + f"{i+1} : {node.name()} {node.type()}\n"
    return str_result

##################################################################################################

def list_files(startpath):
    str_result = ""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = '| ' * (level) + '├ '
        str_result += '{}{}/\n'.format(indent, os.path.basename(root))
        subindent = '| ' * (level + 1) + '├ '
        for f in files:
            str_result += '{}{}\n'.format(subindent, f)
    return str_result


##################################################################################################

def fix_names_as_identifiers(model):
    for node in model.get_nodes():
        name = node.name()
        if not name.isidentifier():
            name = ''.join([c if c.isalnum() else "_" for c in name])
            node.set_name(name)

##################################################################################################

def freeze_producers(model):
    for node in model.get_nodes():
        if node.type() == "Producer":
            #node.get_operator().set_attr("Constant", True)
            node.get_operator().attr.constant = True

##################################################################################################
       
def fix_export(model, export_folder, 
               input_size=5, output_size=5,
               c_type_name="float",
               ctypes_type_name="c_float",
               np_type_name="np.float32",
               input_data_list=[[0,0,0,1,2], [1,2,3,4,5]]):

    filename = 'run.py'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(
f"""
from ctypes import cdll, POINTER, {ctypes_type_name}
import numpy as np
import time
from argparse import ArgumentParser
import json

mylib = cdll.LoadLibrary('./lib/libdnn.so')
mylib.forward.argtypes = [POINTER({ctypes_type_name}), POINTER({ctypes_type_name})]

input_size = {input_size}
input_data_list = {input_data_list}

parser = ArgumentParser(description='Load ACAS-Xu ONNX using AIDGE.')
parser.add_argument("json", nargs='?', type=str, default=None, help="input data file on json format containing just a list of lists")
parser.add_argument("--data", type=str, default=None, help="input data as a list of lists")
args = parser.parse_args()

if args.data is not None:
    input_data_list = json.loads(args.data, parse_float=True)

try:
    if args.json is not None:
        with open(args.json, 'r') as f:
            input_data_list = json.load(f, parse_float=True)
except:
    print("An error happened when trying to read input data from json file.")
 
for input_data in input_data_list:

    result_c = ({ctypes_type_name} * input_size)()
    input_c = ({ctypes_type_name} * input_size)(*input_data)

    #result_np = np.zeros(input_size, dtype={np_type_name})
    #input_np = np.array(input_data, dtype={np_type_name})

    #result_c = result_np.ctypes.data_as(POINTER({ctypes_type_name}))
    #input_c = input_np.ctypes.data_as(POINTER({ctypes_type_name}))

    #get start time of forward execution
    start_time = time.time()
    
    #call DNN forward
    mylib.forward(input_c, result_c)
    
    elapsed_time = (time.time() - start_time) * 1000

    result_np = np.array(np.fromiter(result_c, dtype={np_type_name}, count=input_size))

    print("output:", result_np)
    print("execution time (ms):", elapsed_time)
    
"""
        )

    filename = 'dnn.cpp'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(
f"""
#include "dnn.h"
#include <dnn.hpp>

static const float inputs[{len(input_data_list)}][{input_size}] __attribute__((section(".nn_data"))) = {str(input_data_list).replace('[','{').replace(']','}')};

void forward(const {c_type_name}* input, {c_type_name}* result){{
	model_forward(input, result);
}}
"""
        )

    filename = 'dnn.h'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(
f"""
#ifndef DNN_H
#define DNN_H

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

extern "C" {{
    EXPORT void forward(const {c_type_name}* input, {c_type_name}* result);
}}

#endif // DNN_H
"""
        )

    filename = 'Makefile'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(
"""
CC := g++
CCFLAGS := $(CCFLAGS) -O2 -Wall -Wextra -MMD -fopenmp
LDFLAGS := $(LDFLAGS) -fopenmp

OBJDIR := build
DNNDIR := dnn
BINDIR := bin
LIBDIR := lib

TARGET_STANDALONE := $(BINDIR)/run_export
TARGET_USING_LIB := $(BINDIR)/run
TARGET_SHARED_LIB := $(LIBDIR)/libdnn.so
LINK_OPTIONS := -L$(LIBDIR) -ldnn

$(info ----------------------------------)
$(info OBJDIR is $(OBJDIR))
$(info BINDIR is $(BINDIR))
$(info LIBDIR is $(LIBDIR))

INCLUDE_DIRS :=-I. -I./$(DNNDIR) -I./$(DNNDIR)/include -I./$(DNNDIR)/layers -I./$(DNNDIR)/parameters

CC_SRCS := $(shell find $(DNNDIR) -iname "*.cpp")
CC_OBJS := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CC_SRCS))
DEPENDENCIES := $(patsubst %.o, %.d, $(CC_OBJS))

REQ_OBJS_STANDALONE := $(CC_OBJS) $(OBJDIR)/main.o
REQ_OBJS_USING_LIB  := $(OBJDIR)/main.o
REQ_OBJS_SHARED_LIB := $(OBJDIR)/dnn/src/forward.o $(OBJDIR)/dnn.o

all: build_shared_lib build_exe_using_shared_lib build_exe_standalone end
	
build_exe_standalone: $(REQ_OBJS_STANDALONE)
	$(info ----------------------------------)
	$(info Making .exe)
	@mkdir -p $(BINDIR)
	$(CC) $(REQ_OBJS_STANDALONE) $(LDFLAGS) -o $(TARGET_STANDALONE)

build_exe_using_shared_lib: $(REQ_OBJS_USING_LIB)
	$(info ----------------------------------)
	$(info Making run.exe using shared library dnn.so)
	@mkdir -p $(BINDIR)
	$(CC) $(REQ_OBJS_USING_LIB) $(LDFLAGS) -o $(TARGET_USING_LIB) $(LINK_OPTIONS)

build_shared_lib: $(REQ_OBJS_SHARED_LIB)
	$(info ----------------------------------)
	$(info Making .so)
	@mkdir -p $(LIBDIR)
	$(CC) $(LDFLAGS) $(INCLUDE_DIRS) -fPIC -shared -o $(TARGET_SHARED_LIB) $(REQ_OBJS_SHARED_LIB)
	
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDE_DIRS) -c $< -o $@ 
	
clean:
	$(info Cleaning lib, bin, build folders)
	@if [ -d "$(OBJDIR)" ]; then rm -rf $(OBJDIR); fi
	@if [ -d "$(BINDIR)" ]; then rm -rf $(BINDIR); fi
	@if [ -d "$(LIBDIR)" ]; then rm -rf $(LIBDIR); fi
	$(info ----------------------------------)

end:
	$(info ----------------------------------)
	$(info For running the generated standalone executable:)
	$(info > $(TARGET_STANDALONE))
	$(info For running the generated executable that uses the shared library with dynamic link:)
	$(info > export LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH)
	$(info > $(TARGET_USING_LIB))
	$(info For running the python script that uses the shared library with dynamic link:)
	$(info > python run.py [ json | --data=LIST ] )
	$(info ----------------------------------)

-include $(DEPENDENCIES)
"""
        )
        
    filename = 'main.cpp'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(f"""
#include <iostream>
#include <time.h>
#include "dnn.hpp"
#include "inputs.h"

//------------------------------------------------------------------------

int main(void) {{

    const unsigned int output_size = {output_size};
    const unsigned int input_list_size = {len(input_data_list)};

    {c_type_name} results[output_size];

    //for execution time
    clock_t t;
    double exec_time;
    
	for(unsigned int i=0;i<input_list_size;i++) {{

        //get forward starting time
        t = clock();

        //call DNN forward
        model_forward(inputs[i], results);

        //calculate forward execution elapsed time
        t = clock() - t;
        exec_time = ((double)t)/(CLOCKS_PER_SEC/1000);
        
        //print result
        for (unsigned int j = 0; j < output_size; ++j) {{
            std::cout << j << ": " << results[j] << std::endl;
        }}
        std::cout << "wall execution time:" << exec_time << "ms" << std::endl;
        std::cout << "---------------" << std::endl;

	}}

    return 0;

}}
"""
        )
    
    filename = 'inputs.h'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(f"""
#include <stdint.h>

static const {c_type_name} inputs[{len(input_data_list)}][{input_size}] __attribute__((section(".nn_data"))) = {str(input_data_list).replace('[','{').replace(']','}')};
"""
        )
 
    filename = 'inputs.json'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(json.dumps(input_data_list))
