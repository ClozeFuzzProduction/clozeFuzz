import re
import os
import tqdm
import csv
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
def replace_nth_occurrence(string, old_substring, new_substring, n):
    count = 0
    start_index = 0
    while count < n:
        index = string.find(old_substring, start_index)
        if index == -1:
            break
        count += 1
        if count == n:
            string = string[:index] + new_substring + string[index + len(old_substring):]
        else:
            start_index = index + 1
    return string

def replace_nested_content(string,left_char='{',right_char='}'):
    stack = []
    code_list=[]
    for i, char in enumerate(string):
        if char == left_char:
            stack.append(i)
        elif char == right_char:
            if len(stack) > 0:
                start = stack.pop()
                nested_content = string[start:i+1]
                code_list.append(nested_content)
    block_count = {}
    for block in code_list:
        if block in block_count:
            block_count[block] += 1
        else:
            block_count[block] = 1
    edited_code_list = []
    new_substring = left_char+'<insert>'+right_char
    for code, count in block_count.items():
        for i in range(count):
            modified_string = replace_nth_occurrence(string,code, new_substring,i+1)#string.replace(code, '{<insert>}',i+1)
            edited_code_list.append(modified_string)
    
    return edited_code_list

def compare_text(text1, text2):
    text1 = text1.replace(' ', '').replace('\n', '').replace('\t', '')
    text2 = text2.replace(' ', '').replace('\n', '').replace('\t', '')
    return text1==text2
"""
#Example:

raw_code = '''
#![feature(specialization)]
trait X {
    type U;
    fn f(&self) -> Self::U {
        loop {}
    }
}
impl<T> X for T {
    default type U = ();
}
trait Y {
    fn g(&self) {}
}
impl Y for <() as X>::U {}
impl Y for <i32 as X>::U {}
fn main() {
    ().f().g();
}
'''
code_list = replace_nested_content(raw_code,'(',')')
"""
def data_process():
    left_right=[
        ['{','}'],
        ['(',')'],
        ['[',']'],
        ['<','>'],
    ]
    filepath="/home/dataset/better_format"
    targetpath="/home/dataset/simple_mask"
    files=os.listdir(filepath)

    for file in tqdm.tqdm(files):
        if not file.endswith('.rs'):
            continue
        with open(os.path.join(filepath,file),'r') as f:
            raw_code=f.read()
        flag=0
        for left_char,right_char in left_right:
            # print("_{}_{}_".format(left_char,right_char))
            code_list =replace_nested_content(raw_code,left_char,right_char)
            # print(len(code_list))
            for i,code in enumerate(code_list):
                with open(os.path.join(targetpath,file[:-3]+"-type_"+str(flag)+'_'+str(i)+'.rs'),'w') as f:
                    f.write(code)
            flag+=1
import os
import re
import random
from tqdm import tqdm
import subprocess
import time

from typing import List

import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
VERBOSE =False
model_name = "/home/facebook/incoder-1B"
CUDA=None,
tokenizers_version = tuple(int(n) for n in tokenizers.__version__.split('.'))
if tokenizers_version < (0, 12, 1):
    print("warning: Your tokenizers version looks old and you will likely have formatting issues. We recommend installing tokenizers >= 0.12.1")
kwargs = {}
print("loading model")
model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("loading complete")
if CUDA is  None:
    CUDA=torch.cuda.is_available()
if CUDA:
    model = model.half().cuda()
    print("using cuda")
else:
    print("using cpu")

# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"

def code_infill(code,
                max_to_generate=250, temperature=0.5
                ):
    def make_sentinel(i):
        # signals (1) a location to insert an infill and (2) the start of the infill generation
        return f"<|mask:{i}|>"
    def generate(input: str, max_to_generate: int=128, temperature: float=0.2):
        """
        Do standard left-to-right completion of the prefix `input` by sampling from the model
        """
        input_ids = tokenizer(input, return_tensors="pt").input_ids
        if CUDA:
            input_ids = input_ids.cuda()
        max_length = max_to_generate + input_ids.flatten().size(0)
        if max_length > 2048:
            print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
        with torch.no_grad():
            output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, temperature=temperature, max_length=max_length)
            torch.cuda.empty_cache()
        # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
        detok_hypo_str = tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
        if detok_hypo_str.startswith(BOS):
            detok_hypo_str = detok_hypo_str[len(BOS):]
        return detok_hypo_str
    def infill(parts: List[str], max_to_generate: int=128, temperature: float=0.2, extra_sentinel: bool=True, max_retries: int=1):
        assert isinstance(parts, list)
        retries_attempted = 0
        done = False

        while (not done) and (retries_attempted < max_retries):
            retries_attempted += 1

            if VERBOSE:
                print(f"retry {retries_attempted}")
            
            ## (1) build the prompt
            if len(parts) == 1:
                prompt = parts[0]
            else:
                prompt = ""
                # encode parts separated by sentinel
                for sentinel_ix, part in enumerate(parts):
                    prompt += part
                    if extra_sentinel or (sentinel_ix < len(parts) - 1):
                        prompt += make_sentinel(sentinel_ix)
            
            infills = []
            complete = []

            done = True

            ## (2) generate infills
            for sentinel_ix, part in enumerate(parts[:-1]):
                complete.append(part)
                prompt += make_sentinel(sentinel_ix)
                # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
                completion = generate(prompt, max_to_generate, temperature)
                completion = completion[len(prompt):]
                if EOM not in completion:
                    if VERBOSE:
                        print(f"warning: {EOM} not found")
                    completion += EOM
                    done = False
                completion = completion[:completion.index(EOM) + len(EOM)]
                infilled = completion[:-len(EOM)]
                infills.append(infilled)
                complete.append(infilled)
                prompt += completion
            complete.append(parts[-1])
            text = ''.join(complete)

        if VERBOSE:
            print("generated text:")
            print(prompt)
            print()
            print("parts:")
            print(parts)
            print()
            print("infills:")
            print(infills)
            print()
            print("restitched text:")
            print(text)
            print()
        
        return {
            'text': text, # str, the completed document (with infills inserted)
            'parts': parts, # List[str], length N. Same as passed to the method
            'infills': infills, # List[str], length N-1. The list of infills generated
            'retries_attempted': retries_attempted, # number of retries used (if max_retries > 1)
        } 
    def code_infilling(code,max_to_generate=600, temperature=0.7):
        parts = code.split("<insert>")
        result = infill(parts, max_to_generate=max_to_generate, temperature=temperature)
        # print("completed code:")
        # print(result["text"])
        return result["text"]
    
    return code_infilling(code,max_to_generate=max_to_generate, temperature=temperature)
def compile_rust(filepath,rsfile,opt):
    cmd="rustc {} -C opt-level={}  --out-dir temp".format(rsfile,opt) #--out-dir temp
    time_limit=180
    p=subprocess.Popen(cmd,cwd=filepath, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,text=True)
    try:
        out, err = p.communicate(timeout=time_limit)
        returncode = p.returncode
        if "internal compiler error" in err.lower() or "compiler unexpectedly panicked" in err.lower():
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
def get_err(err):
    
    err_info=re.findall(r"thread 'rustc' panicked at.*?\n",err,re.DOTALL)
    if len(err_info)==0:
        err_info=""
    else:
        err_info=err_info[0]
    start=err.find("query stack during panic:")
    end=err.find("end of query stack")
    stack_info=err[start:end]
    stack_info=stack_info.split("\n")
    stack_info=[info[:info.find("]")+1] for info in stack_info]
    stack_info="\n".join(stack_info)
    return err_info,stack_info
def add_csv(filename,columns,new_line_list):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)

        if file.tell() == 0:
            writer.writerow(columns)

        writer.writerow(new_line_list)
    
filepath="/home/dataset/simple_mask"
targetpath="/home/exam_task_2"
rsfiles=os.listdir(filepath)
df=pd.read_csv("/home/err_info.csv")

for rsfile in tqdm(rsfiles):
    
    if not rsfile.endswith('.rs'):
        continue
    if os.path.exists(os.path.join(targetpath,rsfile)):
        continue
    with open(os.path.join(filepath,rsfile),'r') as f:
        raw_code=f.read()
    
    raw_code_len=len(raw_code)
    
    if len(raw_code)>5000:
        continue
    code=code_infill(raw_code,max_to_generate=300, temperature=0.2)
    if compare_text(raw_code,code):
        continue
    else:
        
        with open(os.path.join(targetpath,rsfile),'w') as f:
            f.write(code)
        csv_file="/home/exam_taskK_2_err_info_new.csv"
        columns=["rsfile","res","err_info","stack_info","flag"]
        res,err=compile_rust(targetpath,rsfile,"0")
        flag=""    
        err_info,stack_info="",""
        if res=="ice":
            err_info,stack_info=get_err(err)  
            flag="same"
            if (err_info=="" or err_info is None or err_info=="\n") and (stack_info=="" or stack_info is None or stack_info=="\n"):
                flag="probalbly same"
            elif err_info not in df["err_info"].values:
                flag="new!"
            elif stack_info not in df["stack_info"].values:
                flag="maybe new!"
        res,err_info,stack_info,flag
        if res!="ok":
            add_csv(csv_file,columns,[rsfile,res,err_info,stack_info,flag])

