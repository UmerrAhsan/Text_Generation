from transformers import Trainer, TrainingArguments
from config import CFG
from transformers import AutoModelForCausalLM


def get_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model




training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2/",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
   # push_to_hub=True,
    save_total_limit=1,
    save_strategy = "epoch",
    load_best_model_at_end = True,

)



def training_function(tokenized_train_dataset, tokenized_val_dataset):
    
    trainer = Trainer(
        model= get_model(CFG['model']['model_name']),
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
    )

     # start the training and display the message after completion
    print("Training is started")
   # trainer.train()
    print("Training is completed")

    return trainer