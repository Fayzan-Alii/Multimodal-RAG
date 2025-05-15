from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
if device == "cuda":
    model = model.half()
model.to(device)
model.eval()

def generate_answer(hits, question):
    # Prepare context from hits (retrieved chunks from PDF docs)
    context = "\n".join([h.get('content', '') for h in hits])

    # --- Few-shot examples from your document domain ---
    few_shot_examples = """
Q: What are the key highlights from the Annual Report 2023-24?
A: To answer this, let's think step by step. First, I will identify the main sections of the report, then summarize the most important achievements and financial outcomes.

Q: According to the FYP Handbook 2023, what are the requirements for project submission?
A: Let's break this down step by step. I will first look for the section on submission requirements, then list the key points mentioned in the handbook.

Q: What financial trends are observed in the financials.pdf document?
A: Let's analyze this step by step. I will review the revenue, expenses, and profit/loss statements, then summarize the main trends and any notable changes.
"""

    # --- Chain-of-Thought (CoT) prompting and prompt assembly ---
    prompt = (
        f"{few_shot_examples}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "A: Let's think step by step."
    )

    # Get model's max input length
    max_input_length = getattr(tokenizer.model_max_length, "real", None) or tokenizer.model_max_length or 1024
    # Tokenize and truncate if needed
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
    attention_mask = torch.ones_like(input_ids)  # Explicit attention mask

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2048,  # Increased answer length
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1,  # Batch size 1
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # Optionally, strip the prompt from the answer
    return answer[len(prompt):].strip() if answer.startswith(prompt) else answer.strip()