

------------------------------------------------
# Code plagiasm detector 


This is a code plagiarism detector that allows you to detect plagiarism between a large corpus of code. The main idea is to calculate embeddings of texts using various methods and predict based on these embeddings using a fully connected neural network

## Installation
The package is tested under Python 3. It can be installed via:
```
git clone https://github.com/tttonyalpha/plagiasm_detector
```

## Usage

Now you need to put all the files in one folder and run the program from source code directory using comand

```bash
python3 ./predict.py path/to/folder/with/input/code


```

## Supported languages
My program employs [tree-sitter](https://tree-sitter.github.io/tree-sitter/) as a backend therefore supports all languages from there 

## References
<a id="1">[1]</a> 
A large-scale computational study of content preservation measures for text style transfer and paraphrase generation <br>
Nikolay Babakov, David Dale, Varvara Logacheva, Alexander Panchenko <br>
[aclanthology.org](https://aclanthology.org/2022.acl-srw.23.pdf)


