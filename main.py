# install the following packages from requirements.txt ( run the following command in the terminal )
# pip install -r requirements.txt


import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import  preprocessing as pre
from config import CFG
import trainer as tr
import visualisation as vis
import testing as test
import inference as inf



# login the hugging face using token
from huggingface_hub import HfApi 

hf_api = HfApi(
    #endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token="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", # Token is not persisted on the machine.
)


if __name__ == '__main__':

    df = pre.read_data(CFG['data_path1'],CFG['data_path2'])               # reading data
    df = pre.preprocess(df)                                               # preprocessing data

    vis.print_max_min_length(df)                                            # visualisation
    vis.plot_histograms_for_individual_lengths(df)

    train_df,val_df,test_df = pre.train_validation_test_split(df,CFG['train_size'],CFG['val_size'],CFG['test_size'])        # splitting data into train, validation and test set

    hf_train_dataset,hf_val_dataset,hf_test_dataset = pre.pandas_to_dataset(train_df,val_df,test_df)                     # converting pandas dataframe to huggingface dataset

    tokenized_train_dataset,tokenized_val_dataset,tokenized_test_dataset = pre.tokenize_hf_dataset(hf_train_dataset,hf_val_dataset,hf_test_dataset)    # tokenizing the dataset
    tokenized_train_dataset,tokenized_val_dataset,tokenized_test_dataset = pre.batch_mapping(tokenized_train_dataset,tokenized_val_dataset,tokenized_test_dataset)   # mapping the dataset into batches


    trainer = tr.training_function(tokenized_train_dataset,tokenized_val_dataset)
    trainer.save_model("./fine_tuned_gpt2_final/")
    trainer.save_state()



    perplexity = test.calculate_perplexity(trainer,tokenized_test_dataset)        # testing the model


    # inference

    model,tokenizer = inf.load_model_tokenizer_for_inference()

    # Test the model with a prompt
    prompt_text = "Her sister Katharina welcomed you withal?"
    responses = inf.generate_response(prompt_text, model,tokenizer)

    for response in responses:
        print(response)


 


