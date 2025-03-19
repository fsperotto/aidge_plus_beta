# -*- coding: utf-8 -*-
"""
Created on 24 June 2024

@author: fperotto
"""

##########################################################
# AIDGE UTILS
##########################################################

import os
import sys
from subprocess import check_output  #  check_call
from importlib import import_module
from importlib.util import find_spec
import json

##########################################################

#import a package or module, installing it if necessary
def require(module_str, package_name=None, source=None):

  if package_name is None:
    package_name = module_str.partition('.')[0]
  #if the package is not installed...
  if find_spec(package_name) is None:
    print(f"Installing {package_name}...")
    #install it using !pip install
    if source is None:
      source = package_name
    log = check_output([sys.executable, '-m', 'pip', 'install', source, '--progress-bar=raw'])
    print(log.decode('ascii'))

  #if the package was not correctly installed... (or the name does not correspond to the source)
  if find_spec(package_name) is None:
    print(f"[ERROR] the package {package_name} is not defined.")  

  #otherwise, import module
  else:

    module = import_module(module_str)
  
    if module is None:
      print(f"[ERROR] the module {module_str} could not be correctly imported.")  

    return module
    
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

#PRINT THE TREE OF FILES AND DIRECTORIES
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

#GIVEN AN AIDGE MODEL, ENSURE THAT ALL THE NODES 
def fix_names_as_identifiers(model):
    for i, node in enumerate(model.get_nodes()):
        name = node.name()
        #if the name is empty
        if name == "":
            node.set_name("NODE_"+str(i))
        #if the name contains invalid characters (it can be a problem when exporting code)
        elif not name.isidentifier():
            #change invalid characters by "_"
            name = ''.join([c if c.isalnum() else "_" for c in name])
            node.set_name(name)

##################################################################################################

#IT SEEMS THAT IT IS FOR EVAL MODE (WITHOUT GRADIENT)
def freeze_producers(model):
    for node in model.get_nodes():
        if node.type() == "Producer":
            #node.get_operator().set_attr("Constant", True)
            node.get_operator().attr.constant = True

##################################################################################################

#MODIFY EXPORT FILES       
def fix_export(model, export_folder, 
               input_size=5, output_size=5,
	           input_name="input",
               c_type_name="float",
               ctypes_type_name="c_float",
               np_type_name="np.float32",
               input_data_list=[[0,0,0,1,2], [1,2,3,4,5]]):

    #--------------------------------------------------------------
    # create RUN.PY
		       
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

		       
    #--------------------------------------------------------------
    # create LIBDNN.CPP    (for lib)
		       
    filename = 'libdnn.cpp'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(
f"""
#include "libdnn.h"
#include "forward.hpp"

void forward(const {c_type_name}* input, {c_type_name}** result){{
	model_forward(input, result);
}}
"""
        )

#static const float inputs[{len(input_data_list)}][{input_size}] __attribute__((section(".nn_data"))) = {str(input_data_list).replace('[','{').replace(']','}')};


    #--------------------------------------------------------------
    # create LIBDNN.H    (for lib)
		       
    filename = 'libdnn.h'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(
f"""
#ifndef LIBDNN_H
#define LIBDNN_H

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

extern "C" {{
    EXPORT void forward(const {c_type_name}* input, {c_type_name}** result);
}}

#endif // LIBDNN_H
"""
        )

    #--------------------------------------------------------------
    # change MAKEFILE
		       
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

TARGET_STANDALONE_OLD := $(BINDIR)/run_old
TARGET_STANDALONE_NEW := $(BINDIR)/run_new
#TARGET_USING_LIB      := $(BINDIR)/run_lib
TARGET_SHARED_LIB     := $(LIBDIR)/libdnn.so
LINK_OPTIONS          := -L$(LIBDIR) -ldnn

$(info ----------------------------------)
$(info OBJDIR is $(OBJDIR))
$(info BINDIR is $(BINDIR))
$(info LIBDIR is $(LIBDIR))

INCLUDE_DIRS :=-I. -I./$(DNNDIR) -I./$(DNNDIR)/include -I./$(DNNDIR)/layers -I./$(DNNDIR)/parameters

CC_SRCS := $(shell find $(DNNDIR) -iname "*.cpp")
CC_OBJS := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CC_SRCS))
DEPENDENCIES := $(patsubst %.o, %.d, $(CC_OBJS))

REQ_OBJS_STANDALONE_OLD := $(CC_OBJS) $(OBJDIR)/main.o
REQ_OBJS_STANDALONE_NEW := $(CC_OBJS) $(OBJDIR)/main_new.o
#REQ_OBJS_USING_LIB  := $(CC_OBJS) $(OBJDIR)/main_lib.o
REQ_OBJS_SHARED_LIB := $(OBJDIR)/dnn/src/forward.o $(OBJDIR)/libdnn.o

#all: build_shared_lib build_exe_using_shared_lib build_exe_standalone end
all: build_shared_lib build_exe_standalone end
	
build_exe_standalone: $(REQ_OBJS_STANDALONE)
	$(info ----------------------------------)
	$(info Making standalone executable files)
	@mkdir -p $(BINDIR)
	$(CC) $(REQ_OBJS_STANDALONE_OLD) $(LDFLAGS) -o $(TARGET_STANDALONE_OLD)
	$(CC) $(REQ_OBJS_STANDALONE_NEW) $(LDFLAGS) -o $(TARGET_STANDALONE_NEW)

#build_exe_using_shared_lib: $(REQ_OBJS_USING_LIB)
#	$(info ----------------------------------)
#	$(info Making executable file that uses shared library libdnn.so)
#	@mkdir -p $(BINDIR)
#	$(CC) $(REQ_OBJS_USING_LIB) $(LDFLAGS) -o $(TARGET_USING_LIB) $(LINK_OPTIONS)

build_shared_lib: $(REQ_OBJS_SHARED_LIB)
	$(info ----------------------------------)
	$(info Making shared library .so)
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
	$(info For running the generated standalone executables:)
	$(info > $(TARGET_STANDALONE_NEW))
	$(info > $(TARGET_STANDALONE_OLD))
	$(info For running the generated executable that uses the shared library with dynamic link:)
	$(info > export LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH)
	$(info > $(TARGET_USING_LIB))
	$(info For running the python script that uses the shared library with dynamic link:)
	$(info > python run.py [ json | --data=LIST ] )
	$(info ----------------------------------)

-include $(DEPENDENCIES)
"""
        )

    #--------------------------------------------------------------
    # create NEW_MAIN.CPP
		       
    filename = 'main_new.cpp'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(f"""
#include <iostream>
#include <time.h>
//#include "dnn.h"
#include "forward.hpp"
#include "inputs.h"

//------------------------------------------------------------------------

int main(void) {{

    const unsigned int output_size = {output_size};
    const unsigned int input_list_size = {len(input_data_list)};

    // Initialize the output arrays
    {c_type_name}* results = nullptr;
    //{c_type_name} results[output_size];

    //for execution time
    clock_t t;
    double exec_time;
    
    for(unsigned int i=0;i<input_list_size;i++) {{

        //get forward starting time
        t = clock();

        // Call the DNN forward function
        model_forward(inputs[i], &results);
        //model_forward(&results);

        //calculate forward execution elapsed time
        t = clock() - t;
        exec_time = ((double)t)/(CLOCKS_PER_SEC/1000);

        // Print the results of each output
        printf("outputs:\\n");
        for (unsigned int j = 0; j < output_size; ++j) {{
            std::cout << j << ": " << results[j] << std::endl;
            //printf("%f ", results[j]);
        }}
        std::cout << "wall execution time:" << exec_time << "ms" << std::endl;
        std::cout << "---------------" << std::endl;

	}}

    return 0;

}}
"""
        )

		       
    #--------------------------------------------------------------
    # create INPUTS.H   in place of {input_name}.h
		       
    filename = 'inputs.h'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(f"""
#include <stdint.h>

static const {c_type_name} inputs[{len(input_data_list)}][{input_size}] __attribute__((section(".nn_data"))) = {str(input_data_list).replace('[','{').replace(']','}')};
"""
        )
 
 
    #--------------------------------------------------------------
    # create INPUTS.JSON

    filename = 'inputs.json'
    with open(os.path.join(export_folder, filename), 'w') as f:
        f.write(json.dumps(input_data_list))

##################################################################################################

#class to convert to equivalent datatypes among different modules
class DTypeConvert:

  valid_types = ['float16', 'float32', 'float64', #'float128',
                 'half', 'single', 'float', 'double',
                 'int2', 'int3', 'int4', 'int5', 'int6', 'int7',
                 'int8', 'int16', 'int32', 'int64',
                 'byte', 'short', 'int', 'long',
                 'uint2', 'uint3', 'uint4', 'uint5', 'uint6', 'uint7',
                 'ubyte', 'ushort', 'uint', 'ulong',
                 'uint8', 'uint16', 'uint32', 'uint64']

  def __init__(self, str_dtype='float32'):

    if str_dtype in self.valid_types:

      #set the datatype
      self._str_dtype = str_dtype

    else:

      print(f'Unknown or unsupported dtype "{str_dtype}". Using "float32".')
      self._str_dtype = 'float32'


  def dtype(self, module:str=None):

    #STRING
    if module is None or module in ['str', 'string']:
    
      return self.str_dtype()
    
    #PYTHON TYPE
    elif module.lower() in ['python', 'py']:
      return {'float16':float, 'float32':float, 'float64':float, #'float128':float,
              'half':float, 'single':float, 'float':float, 'double':float,
              'int2':int, 'int3':int, 'int4':int, 'int5':int, 'int6':int, 'int7':int,
              'int8':int, 'int16':int, 'int32':int, 'int64':int,
              'byte':int, 'short':int, 'int':int, 'long':int,
              'uint2':int, 'uint3':int, 'uint4':int, 'uint5':int, 'uint6':int, 'uint7':int,
              'ubyte':int, 'ushort':int, 'uint':int, 'ulong':int,
              'uint8':int, 'uint16':int, 'uint32':int, 'uint64':int}[self._str_dtype]
    
    #SPECIFIC MODULES
    else:
      
      try:

        #print(module.lower()) 
        
        #NUMPY
        if module.lower() in ['numpy', 'np']:
          import numpy as np
          return {'float16':np.float16, 'float32':np.float32, 'float64':np.float64, #'float128':np.float128,
                  'half':np.half, 'single':np.single, 'float':np.float32, 'double':np.float64,
                  'int2':np.int8, 'int3':np.int8, 'int4':np.int8, 'int5':np.int8, 'int6':np.int8, 'int7':np.int8,
                  'int8':np.int8, 'int16':np.int16, 'int32':np.int32, 'int64':np.int64,
                  'byte':np.byte, 'short':np.short, 'int':np.int32, 'long':np.int64,
                  'uint2':np.uint8, 'uint3':np.uint8, 'uint4':np.uint8, 'uint5':np.uint8, 'uint6':np.uint8, 'uint7':np.uint8,
                  'ubyte':np.ubyte, 'ushort':np.ushort, 'uint':np.uint, 'ulong':np.int64,
                  'uint8':np.uint8, 'uint16':np.uint16, 'uint32':np.uint32, 'uint64':np.uint64}[self._str_dtype]
        
        #AIDGE
        elif module.lower() in ['aidge']:
          import aidge_core
          return {'float16':aidge_core.dtype.float16, 'float32':aidge_core.dtype.float32, 'float64':aidge_core.dtype.float64, #'float128':aidge_core.dtype.float64,
                  'half':aidge_core.dtype.float16, 'single':aidge_core.dtype.float32, 'float':aidge_core.dtype.float32, 'double':aidge_core.dtype.float64,
                  'int2':aidge_core.dtype.int2, 'int3':aidge_core.dtype.int3, 'int4':aidge_core.dtype.int4, 'int5':aidge_core.dtype.int5, 'int6':aidge_core.dtype.int6, 'int7':aidge_core.dtype.int7,
                  'int8':aidge_core.dtype.int8, 'int16':aidge_core.dtype.int16, 'int32':aidge_core.dtype.int32, 'int64':aidge_core.dtype.int64, 
                  'byte':aidge_core.dtype.int8, 'short':aidge_core.dtype.int16, 'int':aidge_core.dtype.int32, 'long':aidge_core.dtype.int64,
                  'uint2':aidge_core.dtype.uint2, 'uint3':aidge_core.dtype.uint3, 'uint4':aidge_core.dtype.uint4, 'uint5':aidge_core.dtype.uint5, 'uint6':aidge_core.dtype.uint6, 'uint7':aidge_core.dtype.uint7,
                  'ubyte':aidge_core.dtype.uint8, 'ushort':aidge_core.dtype.uint16, 'uint':aidge_core.dtype.uint32, 'ulong':aidge_core.dtype.uint64,
                  'uint8':aidge_core.dtype.uint8, 'uint16':aidge_core.dtype.uint16, 'uint32':aidge_core.dtype.uint32, 'uint64':aidge_core.dtype.uint64}[self._str_dtype]

        #TORCH
        elif module.lower() in ['torch', 'pytorch', 'pt']:
          import torch
          return {'float16':torch.float16, 'float32':torch.float32, 'float64':torch.float64, #'float128':torch.float64,
                  'half':torch.half, 'single':torch.float, 'float':torch.float, 'double':torch.double,
                  'int2':torch.int8, 'int3':torch.int8, 'int4':torch.int8, 'int5':torch.int8, 'int6':torch.int8, 'int7':torch.int8,
                  'int8':torch.int8, 'int16':torch.int16, 'int32':torch.int32, 'int64':torch.int64,
                  'byte':torch.int8, 'short':torch.short, 'int':torch.int, 'long':torch.long,
                  'uint2':torch.uint8, 'uint3':torch.uint8, 'uint4':torch.uint8, 'uint5':torch.uint8, 'uint6':torch.uint8, 'uint7':torch.uint8,
                  'ubyte':torch.uint8, 'ushort':torch.uint16, 'uint':torch.uint32, 'ulong':torch.uint64,
                  'uint8':torch.uint8, 'uint16':torch.uint16, 'uint32':torch.uint32, 'uint64':torch.uint64}[self._str_dtype]
  
        #TF
        elif module.lower() in ['ft', 'tensorflow']:
          import tensorflow as tf
          return {'float16':tf.float16, 'float32':tf.float32, 'float64':tf.float64, #'float128':tf.float64,
                        'half':tf.half, 'single':tf.float32, 'float':tf.float32, 'double':tf.double,
                        'int2':tf.int8, 'int3':tf.int8, 'int4':tf.int8, 'int5':tf.int8, 'int6':tf.int8, 'int7':tf.int8,
                        'int8':tf.int8, 'int16':tf.int16, 'int32':tf.int32, 'int64':tf.int64,
                        'byte':tf.int8, 'short':tf.int16, 'int':tf.int32, 'long':tf.int64,
                        'uint2':tf.uint8, 'uint3':tf.uint8, 'uint4':tf.uint8, 'uint5':tf.uint8, 'uint6':tf.uint8, 'uint7':tf.uint8,
                        'ubyte':tf.uint8, 'ushort':tf.uint16, 'uint':tf.uint32, 'ulong':tf.uint64,
                        'uint8':tf.uint8, 'uint16':tf.uint16, 'uint32':tf.uint32, 'uint64':tf.uint64}[self._str_dtype]
  
        #C_TYPES
        elif module.lower() in ['ctypes', 'c-types', 'c_types']:
          import ctypes
          return {'float16':ctypes.c_float, 'float32':ctypes.c_float, 'float64':ctypes.c_double, #'float128':ctypes.c_longdouble,
                  'half':ctypes.c_float, 'single':ctypes.c_float, 'float':ctypes.c_float, 'double':ctypes.c_double, #'float128':ctypes.c_longdouble,
                  'int2':ctypes.c_int8, 'int3':ctypes.c_int8, 'int4':ctypes.c_int8, 'int5':ctypes.c_int8, 'int6':ctypes.c_int8, 'int7':ctypes.c_int8,
                  'int8':ctypes.c_int8, 'int16':ctypes.c_int16, 'int32':ctypes.c_int32, 'int64':ctypes.c_int64,
                  'byte':ctypes.c_byte, 'short':ctypes.c_short, 'int':ctypes.c_int, 'long':ctypes.c_long,
                  'uint2':ctypes.c_uint8, 'uint3':ctypes.c_uint8, 'uint4':ctypes.c_uint8, 'uint5':ctypes.c_uint8, 'uint6':ctypes.c_uint8, 'uint7':ctypes.c_uint8,
                  'ubyte':ctypes.c_ubyte, 'ushort':ctypes.c_ushort, 'uint':ctypes.c_uint, 'ulong':ctypes.c_ulong,
                  'uint8':ctypes.c_uint8, 'uint16':ctypes.c_uint16, 'uint32':ctypes.c_uint32, 'uint64':ctypes.c_uint64}[self._str_dtype]
        
        else:
          print(f'Unknown module "{module}".')
          return None
          
      except:
        print(f'Unavailable module "{module}".')
        return None


  def str_dtype(self):
    return self._str_dtype
    
  def __str__(self):
    return f"DType('{self._str_dtype}')"
