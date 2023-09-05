'''
This is the configuration file for the project.
if you want to change any of the parameters, you can change it here. We will change the model and tokenizer in this file.
'''
CFG = {

    'data_path1': 'Dataset\ROCStories__spring2016 - ROCStories_spring2016.csv' ,
    'data_path2': 'Dataset\ROCStories_winter2017 - ROCStories_winter2017.csv',
    'train_size': 0.7,
    'val_size': 0.1,
    'test_size': 0.2,

    'tokenizer': {
        'tokenizer_type': 'AutoTokenizer',
        'tokenizer_name': 'distilgpt2',
    },



    'model': {
        'model_type': 'AutoModelForCausalLM',
        'model_name': 'distilgpt2'
    },
    'trainer': {
        'output_dir': 'weights',
        'evaluation_strategy': 'epoch',
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'num_train_epochs': 1,
        'weight_decay': 0.01,
        'push_to_hub': True
    },
    'inference': {
        'inferece_task': 'Text_Generation',
        'model_path': 'weights'
    }

    }
