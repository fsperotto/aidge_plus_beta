# -*- coding: utf-8 -*-
"""
Created on 24 June 2024

@author: fperotto
"""

##########################################################
# ONERA PROXY UTILS
##########################################################

import platform    # For getting the operating system name
import subprocess  # For executing a shell command

##########################################################
# functions for proxy and onera proxy
##########################################################

def unset_proxy():
    set_proxy(http_proxy='', https_proxy='' , no_proxy=''):

#---------------------------------------------------------

def verify_and_set_proxy_onera():

    # Option for the number of packets as a function of
    #param = '-n' if os.sys.platform().lower()[:3]=='win' else '-c'
    param = '-n' if if platform.system().lower()[:3]=='win' else '-c'
    hostname = "proxy.onera"
    # Building the command. Ex: "ping -c 1 proxy.onera"
    command = ['ping', param, '1', host]
    #response = os.system(f"ping {param} 1 {hostname}")  #os system call is vulnerable to injection
    response = subprocess.call(command)    
    if response == 0:
        set_proxy(http_proxy='http://proxy.onera:80', 
                  https_proxy='https://proxy.onera:80', 
                  no_proxy='localhost, 127.0.0.0/8, *.onera.net, *.onera'):
    else:
        print("Proxy not available.")

#---------------------------------------------------------

def verify_and_set_proxy_onecert():
    # Option for the number of packets as a function of
    #param = '-n' if os.sys.platform().lower()[:3]=='win' else '-c'
    param = '-n' if if platform.system().lower()[:3]=='win' else '-c'
    hostname = "proxy.onera"
    # Building the command. Ex: "ping -c 1 proxy.onera"
    command = ['ping', param, '1', host]
    #response = os.system(f"ping {param} 1 {hostname}")  #os system call is vulnerable to injection
    response = subprocess.call(command)    
    if response == 0:
        set_proxy(http_proxy='http://proxy.onecert.fr:80', 
                  https_proxy='https://proxy.onecert.fr:80', 
                  no_proxy='localhost, 127.0.0.0/8, *.onera.net, *.onera'):
    else:
        print("Proxy not available.")
  
#---------------------------------------------------------

def set_proxy(http_proxy=None, https_proxy=None , no_proxy=None):
    os.environ['NO_PROXY'] = no_proxy
    os.environ['HTTP_PROXY'] = http_proxy
    os.environ['HTTPS_PROXY'] = https_proxy
