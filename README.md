# Scientific_Paper_App
🤖 Scientific Article Assistant
This project demonstrates how to fine-tune a large language model (LLaMA 3 8B) on scientific paper data to create a specialized assistant that can summarize research articles or answer questions about them. It includes both training logic and a Gradio-powered web app for interactive use.

🔧 Project Structure
bash
复制代码
.
├── .env                         # Stores Hugging Face token
├── .gitignore                  # Ignore checkpoints, virtualenv, etc.
├── app.py                      # Entry point for launching the Gradio app
├── README.md                   # Project documentation
├── notebooks/
│   ├── finetuning.ipynb 
    ├── sampleArticle.txt        # Example input article for app testing
📘 What Each File Does
notebooks/finetuning.ipynb
This Jupyter notebook walks through:

Loading and preprocessing scientific paper datasets.

Formatting the instruction dataset in ChatML-style JSONL.

Fine-tuning a LLaMA-3 8B base model using LoRA with trl.SFTTrainer.

Pushing the model to Hugging Face Hub.

👉 The fine-tuned model is available at:
📍 Benny97/ScientificPaperLLMs-2025-05-10_22.43.03

app.py
This script launches a Gradio web app that:

Lets you paste or upload a .txt article.

Provides two modes:

Summarize: Generates a concise summary.

Chat: Lets you ask questions based on the article.

Runs locally at http://127.0.0.1:7860/

sampleArticle.txt
A sample long article (copied from the training domain) to test the summarization and Q&A modes of your assistant.

🚀 How to Run This Project
1. Clone the repository & set up a virtual environment:

    git clone https://github.com/yourname/ScientificPaperLLMs.git
    cd ScientificPaperLLMs
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
2. Install dependencies:

    pip install -r requirements.txt
3. Configure Hugging Face token
Create a .env file in the root directory and add:


    HUGGINGFACE_RW_TOKEN=your_token_here
🔐 You can find your token here: https://huggingface.co/settings/tokens

4. Launch the Gradio app:

    python app.py
    Then open your browser and visit: http://127.0.0.1:7860

📦 Model Info
Model	Size	Quantization	Format	Hosted At
LLaMA-3 8B (LoRA)	8B	4-bit	PEFT	Benny97/ScientificPaperLLMs-2025-05-10_22.43.03