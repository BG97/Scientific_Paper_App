import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,set_seed
from trl import SFTTrainer, SFTConfig
from random import seed as set_seed
from huggingface_hub import login
import os
from dotenv import load_dotenv
# ==================== HFLogin ====================
load_dotenv(override=True)
os.environ['HUGGINGFACE_RW_TOKEN'] = os.getenv('HUGGINGFACE_RW_TOKEN')
hf_token = os.environ['HUGGINGFACE_RW_TOKEN']
login(hf_token, add_to_git_credential=True)

# ==================== CONFIGURATION ====================
HF_USER = "Benny97"
RUN_NAME = "2025-05-10_22.43.03"
PROJECT_NAME = "ScientificPaperLLMs"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

print(f"[INFO] Loading tokenizer from base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"[INFO] Loading base model...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map='cuda'
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

print(f"[INFO] Loading LoRA fine-tuned model: {FINETUNED_MODEL}")
fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL).to("cuda")

# ==================== GENERATION FUNCTION ====================
def model_Gen(prompt):
    set_seed(42)
    if "### Response:" in prompt:
        prompt = prompt.split("### Response:")[0] + "### Response:"

    print(f"[INFO] Generating for prompt of length {len(prompt)} tokens...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = fine_tuned_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        do_sample=False,
        num_return_sequences=1,
    )
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

# ==================== PROMPT TEMPLATES ====================
def generate_response(article, user_message, mode):
    if not article:
        return "Please upload or input an article."
    if mode == "Summarize":
        prompt = f"""### Instruction:
                    Summarize the following article.
                    
                    ### Article:
                    {article}
                    
                    ### Response:"""
    else:
        if not user_message:
            return "Please enter a message to chat about the article."
        prompt = f"""### Instruction:
                    You are a helpful assistant. Answer the question below based on the article.
                    
                    ### Article:
                    {article}
                    
                    ### Question:
                    {user_message}
                    
                    ### Response:"""

    with torch.no_grad():
        return model_Gen(prompt)

# ==================== FILE INPUT HANDLER ====================
def process_file(file):
    with open(file.name, "r", encoding="utf-8") as f:
        return f.read()

# ==================== CLEAR INPUT ====================
def clear_article():
    return ""

# ==================== GRADIO UI ====================
def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 Scientific Article Assistant")

        mode = gr.Radio(["Summarize", "Chat"], value="Summarize", label="Mode")
        article_input = gr.Textbox(lines=15, label="Paste Article or Upload File")
        file_upload = gr.File(label="Or Upload a .txt File", file_types=[".txt"])
        clear_btn = gr.Button("🗑️ Clear Article Input", variant="secondary")
        user_message = gr.Textbox(lines=2, label="Your Question (for Chat mode)")
        output = gr.Textbox(label="Model Response")
        generate_btn = gr.Button("Generate")

        clear_btn.click(fn=clear_article, outputs=article_input)
        file_upload.change(fn=process_file, inputs=file_upload, outputs=article_input)
        generate_btn.click(
            fn=generate_response,
            inputs=[article_input, user_message, mode],
            outputs=output,
        )

    demo.launch()
# ==================== MAIN ====================
if __name__ == "__main__":
    launch_app()
