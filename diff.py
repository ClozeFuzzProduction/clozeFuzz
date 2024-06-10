# differencial testing for gcc and llvm
import os
import re
import random
import logging
from tqdm import tqdm
import subprocess
import time

import pandas as pd


def get_c_files(directory,suffix=['c','cpp','cc','hpp']):
    c_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1] in suffix:
                c_files.append(os.path.join(root, file))
    return c_files

def compile_llvm(filepath,file,opt):
    
    cmd="clang++ {} -O{} -o code".format(file,opt) #--out-dir temp
    
    time_limit=60
    p=subprocess.Popen(cmd,cwd=filepath, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,text=True)
    try:
        
        out, err = p.communicate(timeout=time_limit)
        returncode = p.returncode
        if returncode!=0:
            return "error",err
    except subprocess.TimeoutExpired:
        p.terminate()
        return "timeout","" 

    
    cmd="./code"
    p=subprocess.Popen(cmd,cwd=filepath, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,text=True)
    
    try:
        out, err = p.communicate(timeout=time_limit)
        returncode = p.returncode
        if returncode!=0:
            return "error",err
    except subprocess.TimeoutExpired:
        p.terminate()
        return "timeout","" 

    return "ok",out

def compile_gcc(filepath,file,opt):
    
    cmd="g++ {} -O{} -o code".format(file,opt) 
    
    time_limit=60
    p=subprocess.Popen(cmd,cwd=filepath, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,text=True)
    try:
        
        out, err = p.communicate(timeout=time_limit)
        returncode = p.returncode
        if returncode!=0:
            return "error",err
    except subprocess.TimeoutExpired:
        p.terminate()
        return "timeout","" 

    
    cmd="./code"
    p=subprocess.Popen(cmd,cwd=filepath, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,text=True)
    
    try:
        out, err = p.communicate(timeout=time_limit)
        returncode = p.returncode
        if returncode!=0:
            return "error",err
    except subprocess.TimeoutExpired:
        p.terminate()
        return "timeout","" 

    return "ok",out


def compile_diff_llvm(filepath,file):
    
    llvm_opts=['0','1','2','3','s','z']
    dict_res={}
    
    for llvm_opt in llvm_opts:
        llvm_res,llvm_out=compile_llvm(filepath,file,llvm_opt)
        dict_res["llvm_"+llvm_opt]=[llvm_res,llvm_out]
    
    res_list=[dict_res[key][0] for key in dict_res]
    if len(set(res_list))==1:
        return "ok",dict_res
    else:
        return "error",dict_res



if __name__=="__main__":
    
    cc_files=get_c_files("/home/cTest/gcc_llvm_mask")
    
    for file in tqdm(cc_files):
        filename=file.split('/')[-1]
        filepath="/".join(file.split('/')[:-1])
        
        res,res_dict=compile_diff_llvm(filepath,filename)
        
        if os.path.exists(filepath+"/code"):
            os.remove(filepath+"/code")
        if res!="ok":
            
            with open('/home/cTest/gcc_llvm_compile_diff.csv','a') as f:
                f.write("{},{},{}\n".format(file,res,res_dict))
        