import math

def calculate_perplexity(trainer, tokenized_test_dataset):
    eval_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

   # eval_results2 = trainer.evaluate()
   # print(f"Perplexity: {math.exp(eval_results2['eval_loss']):.2f}")



