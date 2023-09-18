

------------------------------------------------
# Code plagiasm detector 
![image](https://github.com/tttonyalpha/plagiasm_detector/assets/79598074/8491129c-15ef-4a5d-9430-60602e980e36)


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
