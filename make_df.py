import os
import torch
import argparse
import pandas as pd
import code_tokenize as ctok
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import LongformerModel, AutoTokenizer


def process_files(folder_path):
    file_contents = []
    file_names = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                content = f.read()
                file_contents.append(content)
                file_names.append(file)
    return (file_contents, file_names)


def code_tokenizer(s, mode):
    return list(map(lambda x: x.text, list(filter(lambda x: x.type != 'comment', ctok.tokenize(s, lang=mode)))))


def add_tf_idf(corpus, df, MODE='python'):

    vectorizer = TfidfVectorizer(tokenizer=lambda s: code_tokenizer(s, MODE))
    data = vectorizer.fit_transform(corpus).toarray()
    print(data.shape)
    #feature_names = vectorizer.get_feature_names()
    #df[feature_names] = pd.DataFrame(data, index=df.index)
    df['tf_idf_emb'] = data.tolist()

    return df


def add_emb(file_contents, df, model, tokenizer):

    emb_list = []

    for inpt in file_contents:
        tokens = tokenizer.tokenize(inpt)
        marked_tokens = ['[CLS]'] + tokens + ['[SEP]']

        indexed_tokens = tokenizer.convert_tokens_to_ids(marked_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            emb = model(tokens_tensor).pooler_output[0]

        emb_list.append(emb.numpy())

    df['embeddings'] = emb_list

    return df


def make_df(mode='python')
    '''
    mode: programming language - one of javascript, go, ruby, cpp, c, c++, swift, rust, python
    '''
    parser = argparse.ArgumentParser(
        description='Process files in a directory and save their vector representations to a CSV file.')
    parser.add_argument('folder_path', type=str,
                        help='Path to the folder containing the files')
    args = parser.parse_args()

    folder_path = args.folder_path
    if not os.path.isdir(folder_path):
        print('Invalid folder path.')
    else:

        file_contents, file_names = process_files(folder_path)
        df = pd.DataFrame({'file': file_names})

        model_longformer = LongformerModel.from_pretrained(
            "allenai/longformer-base-4096")
        tokenizer_longformer = AutoTokenizer.from_pretrained(
            "allenai/longformer-base-4096")

        add_tf_idf(file_contents, df, mode)
        add_emb(file_contents, df, model_longformer, tokenizer_longformer)

    return df
