from transformers import pipeline, set_seed

# small model for speed
generator = pipeline("text-generation", model="distilgpt2")

def generate_text(user_prompt, sentiment, max_length=150):
    # sentiment: 'positive'|'negative'|'neutral'
    instruction = f"Write a {sentiment} paragraph about: "
    full_prompt = instruction + user_prompt
    out = generator(full_prompt, max_length=max_length, do_sample=True,
                    top_k=40, top_p=0.9, num_return_sequences=1)
    return out[0]['generated_text']
