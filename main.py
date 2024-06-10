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

from IncoderModel import InCoder
from masking import ClozeMask

# 设置环境变量：TOKENIZERS_PARALLELISM=false
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_c_files(directory,suffix=['c','cpp','cc','hpp']):
    c_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1] in suffix:
                c_files.append(os.path.join(root, file))
    return c_files

def compile_llvm(filepath,file,opt):
    
    cmd="clang {} -O{}".format(file,opt) #--out-dir temp
    
    time_limit=60
    p=subprocess.Popen(cmd,cwd=filepath, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,text=True)
    try:
        out, err = p.communicate(timeout=time_limit)
        returncode = p.returncode
        if "stack dump"  in err.lower():
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
    # 开启日志
    logging.basicConfig(level=logging.INFO #设置日志输出格式
                    ,filename="demo.log" #log日志输出的文件位置和文件名
                    ,filemode="w" #文件的写入格式，w为重新写入文件，默认是追加
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/facebook/incoder-1B"
    tokenizer_path = "/home/facebook/incoder-1B"
    incoder = InCoder(model_path, tokenizer_path, device)

    opts=['0','1','2','3','s','z']
    csv_file = '/home/cTest/gcc_llvm_compile_result.csv'

    c_files=get_c_files('/home/cTest/gcc_llvm')
    logging.info('============================\n c_files num:{}\n=========================\n'.format(len(c_files)))
    cloze_mask = ClozeMask()
    step=0
    skip=0
    for c_file in tqdm(c_files):
        step+=1
        if step<645:
            continue
        with open(c_file,'r',errors='ignore') as f:
            code = f.read()
        if len(code)>200:
            skip+=1
            continue
        masked_codes = cloze_mask.mask_singel_code(code)
        cnt=0
        for masked_code in masked_codes:
            cnt+=1
            masked_file = c_file.replace('gcc_llvm','gcc_llvm_mask')
            filename=c_file.split('/')[-1]
            newfilename=filename.split('.')[0]+'_'+str(cnt)+'.'+filename.split('.')[1]
            masked_file=masked_file.replace(filename,newfilename)
            if not os.path.exists(os.path.dirname(masked_file)):
                os.makedirs(os.path.dirname(masked_file))
            new_code=incoder.code_infilling(masked_code)
            with open(masked_file,'w') as f:
                f.write(new_code)

            for opt in opts:
                res,err=compile_llvm(os.path.dirname(masked_file),os.path.basename(masked_file),opt)
                
                if(step%100==0 and cnt%5==0):
                    logging.info('step:{}\nfile:{}\nmasked_file:{}\nopt:{}\nres:{}\nskip:{}\n'.format(step,c_file,masked_file,opt,res,skip) )  
                if res!="ok":
                    # 记录错误信息
                    with open(csv_file,'a') as f:
                        f.write("{},{},{},{}\n".format(masked_file,opt,res,err))

        
        





