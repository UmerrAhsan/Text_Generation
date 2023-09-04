from transformers import AutoModelForCausalLM, AutoTokenizer
from config import CFG


def load_model_tokenizer_for_inference():
    model = AutoModelForCausalLM.from_pretrained("./fine_tuned_gpt2_final/")
    tokenizer = AutoTokenizer.from_pretrained(CFG['tokenizer']['tokenizer_name'])
    return model,tokenizer


# Function to generate responses
def generate_response(prompt_text, model,tokenizer, max_length=55, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # Generate response
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
       # temperature=0.7,
        #top_p=0.9,
    )

    # Decode the generated responses
    responses = []
    for response_id in output_sequences:
        response = tokenizer.decode(response_id, skip_special_tokens=True)
        responses.append(response)

    return responses