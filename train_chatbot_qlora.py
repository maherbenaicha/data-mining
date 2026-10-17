"""
Script complet de Fine-Tuning QLoRA avec Unsloth & Hugging Face TRL.
Idéal pour fine-tuner un modèle de code (ex. Qwen 2.5 Coder 7B ou Llama 3.1 8B).
Fonctionne sur GPU local (RTX 3060/4060/etc.) ou sur Google Colab T4/A100.
"""

import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration du modèle cible
# Qwen/Qwen2.5-Coder-7B-Instruct est le modèle le plus performant pour Python/PostGIS
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 2048  # Le code SIG Python est long
OUTPUT_DIR = "./lora_model_dgre"

def main():
    print("🚀 Chargement d'Unsloth pour Fine-Tuning ultra-rapide (2x plus rapide, -70% VRAM)...")
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        print("⚠️ Unsloth non détecté. Passage en mode standard HuggingFace PEFT / bitsandbytes.")
        use_unsloth = False

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,  # 4-bit QLoRA
            dtype=None,
        )

        # Configuration des adaptateurs LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Rang LoRA recommandé (16 ou 32)
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,  # Optimisé pour Unsloth (0 = plus rapide)
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        # Fallback classique Hugging Face
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # Chargement des données découpées
    dataset = load_dataset("json", data_files={"train": "train_split.jsonl", "test": "test_split.jsonl"})

    # Formatage du Chat Template (system -> user -> assistant)
    def formater_prompts(batch):
        textes = []
        for conv in batch["messages"]:
            prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            textes.append(prompt)
        return {"text": textes}

    dataset = dataset.map(formater_prompts, batched=True)

    # Paramètres d'entraînement
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Batch effectif = 8
        warmup_steps=10,
        max_steps=120,  # Environ 1 à 2 époques complètes
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=60,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none"
    )

    # Lancement du Trainer TRL
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("\n🏋️ Début du Fine-Tuning...")
    trainer.train()

    print(f"\n💾 Sauvegarde de l'adaptateur LoRA dans {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Fine-Tuning terminé avec succès !")

if __name__ == "__main__":
    main()
