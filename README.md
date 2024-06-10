# clozeFuzz
## Introduction
clozeFuzz is a novel fuzzing tool that leverages large language models (LLMs) to generate effective test cases for compilers. The key idea behind clozeFuzz is to identify the bracket structure of code and use it to guide the generation of new test cases through masked token completion.
### Kye Features
- **Bracket Structure Identification**: clozeFuzz analyzes the code's bracket structure to understand the control flow and syntactic constraints of the program under test.
-** Masked Token Completion**: By masking tokens in the code and using an LLM to predict the missing tokens, clozeFuzz can generate new test cases that are syntactically valid and semantically meaningful.
- **Compiler Bug Discovery**: clozeFuzz has already discovered over 30 bugs in Rust and C compilers, which have been confirmed by the respective developer communities.
- **Simple Deployment**: clozeFuzz is designed to be easy to deploy, with minimal setup required. It can be integrated into existing CI/CD pipelines to continuously test compilers.
## Bug found by our tool
You can see more details in our [bug report](https://github.com/ClozeFuzzProduction/clozeFuzz/blob/master/bug.md)
