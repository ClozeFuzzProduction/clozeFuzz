import os
import re
import random
import logging
from tqdm import tqdm
import subprocess
import time

import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def get_c_files(directory,suffix=['c','cpp','cc','hpp']):
    c_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1] in suffix:
                c_files.append(os.path.join(root, file))
    return c_files

def compile_gcc(filepath,file,opt):
    
    cmd="g++ {} -O{}".format(file,opt) #--out-dir temp
    
    time_limit=60
    p=subprocess.Popen(cmd,cwd=filepath, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,text=True)
    try:
        out, err = p.communicate(timeout=time_limit)
        returncode = p.returncode
        if "stack dump"  in err.lower() or "internal compiler error" in err.lower():
            return "ice",err
        elif returncode==137:
            return "mem err",err
        elif "process didn't exit successfully" in err.lower():
            return "crash",err
        else:
            return "ok",err
    except subprocess.TimeoutExpired:
        p.terminate()
        return "timeout","" 
    
if __name__=="__main__":
    
    logging.basicConfig(level=logging.INFO 
                    ,filename="demo.log" 
                    ,filemode="w" 
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" 
                    
                    ,datefmt="%Y-%m-%d %H:%M:%S" 
                    )
    opts=['0','1','2','3','fast']
    csv_file = '/home/cTest/gcc_llvm_compile_result.csv'
    step=0
    cc_files=get_c_files("/home/cTest/gcc_llvm_mask")
    for file in tqdm(cc_files):
        filename=file.split('/')[-1]
        step+=1
        filepath="/".join(file.split('/')[:-1])
        for opt in opts:
            res,err=compile_gcc(filepath,filename,opt)
            
            if(step%100==0):
                logging.info('step:{}\nfile:{}\nmasked_file:{}\nopt:{}\nres:{}\n\n'.format(step,file,file,opt,res) )  
            if res!="ok":
                
                with open(csv_file,'a') as f:
                    f.write("{},{},{},{}\n".format(file,opt,res,err))
            

        
        





