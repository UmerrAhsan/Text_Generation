import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from config import CFG


def read_data(path1,path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df = pd.concat([df1,df2])
    return df
   
   

def preprocess(df):
    df = df[['storyid','storytitle','sentence1','sentence2','sentence3','sentence4','sentence5']]
    df['text'] = df['sentence1'] + df['sentence2'] + df['sentence3'] + df['sentence4'] + df['sentence5'] 
    df = df[['text']]
    return df


def train_validation_test_split(df,training_proportion,validation_proportion,testing_proportion):

    size = len(df)

    training_size = int(training_proportion*size)
    validation_size = int(validation_proportion*size)


    train_df = df[0:training_size]
    val_df = df[training_size:training_size+validation_size]
    test_df = df[training_size+validation_size:]
    return train_df,val_df,test_df


def pandas_to_dataset(df1,df2,df3):
    hf_train_dataset = Dataset.from_pandas(df1)
    hf_val_dataset = Dataset.from_pandas(df2)
    hf_test_dataset = Dataset.from_pandas(df3)
    return hf_train_dataset,hf_val_dataset,hf_test_dataset





def get_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return tokenizer


def tokenize_function(examples):
    tokenizer = get_tokenizer(CFG['tokenizer']['tokenizer_name'])
    return tokenizer(examples["text"])


def tokenize_hf_dataset(hf_train_dataset,hf_val_dataset, hf_test_dataset):
    tokenized_train_dataset = hf_train_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    tokenized_val_dataset = hf_val_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"] )
    tokenized_test_dataset = hf_test_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["__index_level_0__"])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["__index_level_0__"])
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["__index_level_0__"])
    return tokenized_train_dataset,tokenized_val_dataset,tokenized_test_dataset




def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
   # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    concatenated_examples = {k: sum([x if isinstance(x, list) else [x] for x in examples[k]], []) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def batch_mapping(tokenized_train_dataset,tokenized_val_dataset,tokenized_test_dataset):

    tokenized_train_dataset = tokenized_train_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    tokenized_test_dataset = tokenized_test_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    tokenized_val_dataset = tokenized_val_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    return tokenized_train_dataset,tokenized_val_dataset,tokenized_test_dataset