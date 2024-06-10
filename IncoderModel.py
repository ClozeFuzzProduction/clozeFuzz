# 模型InCoder的实现
import os
import re
import random
import logging
from tqdm import tqdm
import subprocess
import time

from typing import List

import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path, tokenizer_path,device ):
    kwargs = {}
    logging.info(f"loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    model = model.half().to(device)
    logging.info(f"loading tokenizer from {tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


class InCoder:
    def __init__(self, model_path, tokenizer_path, device):
        self.model, self.tokenizer = load_model(model_path, tokenizer_path, device)
        self.device = device
        self.BOS = "<|endoftext|>"
        self.EOM = "<|endofmask|>"

    def make_sentinel(self,i):
        return f"<|mask:{i}|>"
    
    def generate(self,  input,max_to_generate=128,temperature=0.2):
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        max_length = max_to_generate + input_ids.flatten().size(0)
        if max_length > 2048:
            logging.warning("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
        with torch.no_grad():
            output = self.model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, temperature=temperature, max_length=max_length)
            torch.cuda.empty_cache()
        # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
        detok_hypo_str = self.tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
        if detok_hypo_str.startswith(self.BOS):
            detok_hypo_str = detok_hypo_str[len(self.BOS):]
        return detok_hypo_str
        

    def infill(self,parts,max_to_generate=128,temperature=0.2,extra_sentinel=True,max_retries=1):
        assert isinstance(parts, list)
        retries_attempted = 0
        done = False

        while (not done) and (retries_attempted < max_retries):
            retries_attempted += 1

            
            
            ## (1) build the prompt
            if len(parts) == 1:
                prompt = parts[0]
            else:
                prompt = ""
                # encode parts separated by sentinel
                for sentinel_ix, part in enumerate(parts):
                    prompt += part
                    if extra_sentinel or (sentinel_ix < len(parts) - 1):
                        # print("sentinel_ix:",sentinel_ix)
                        # prompt += self.make_sentinel(sentinel_ix)
                        prompt+=f"<|mask:{sentinel_ix}|>"
            
            infills = []
            complete = []

            done = True

            ## (2) generate infills
            for sentinel_ix, part in enumerate(parts[:-1]):
                complete.append(part)
                # prompt += self.make_sentinel(sentinel_ix)
                prompt+=f"<|mask:{sentinel_ix}|>"
                # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
                completion = self.generate(prompt, max_to_generate, temperature)
                completion = completion[len(prompt):]
                if self.EOM not in completion:
                    
                    completion += self.EOM
                    done = False
                completion = completion[:completion.index(self.EOM) + len(self.EOM)]
                infilled = completion[:-len(self.EOM)]
                infills.append(infilled)
                complete.append(infilled)
                prompt += completion
            complete.append(parts[-1])
            text = ''.join(complete)

        return {
            'text': text, 
            'parts': parts, 
            'infills': infills, 
            'retries_attempted': retries_attempted, 
        } 
    
    def code_infilling(self,maskedCode,max_to_generate=128,temperature=0.2):
        parts = maskedCode.split("<insert>")
        result = self.infill(parts, max_to_generate=max_to_generate, temperature=temperature)
       
        return result["text"]



if __name__=="__main__":
    logging.basicConfig(level=logging.INFO 
                    ,filename="demo.log" 
                    ,filemode="w" 
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" 
                    
                    ,datefmt="%Y-%m-%d %H:%M:%S" 
                    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/facebook/incoder-1B"
    tokenizer_path = "/home/facebook/incoder-1B"
    incoder = InCoder(model_path, tokenizer_path, device)
    maskedCode = '''
#include <stdio.h>
#include <stdint.h>
int16_t g;
int32_t f;
uint32_t h;
uint64_t j = 6;
static uint64_t o(int16_t x, uint32_t *a, int16_t ac) {
  if (<insert>) j |= f;
  f = ac ? 7 % ac : 7;
  return j;
}
int main() {
  uint64_t v1 =  o(g, &h, 8);
  uint64_t v2 = o(v1, &h, 0);
  
  printf("%lu\n", j);
  return 0;
}

'''
    ans=incoder.code_infilling(maskedCode)
    logging.info(ans)
