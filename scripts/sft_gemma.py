import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer  # type: ignore

MODEL_ID = "/opt/gpudata/models/google/gemma-4-31B-it"
DATASET_ID = "/opt/gpudata/medical-o1-reasoning-SFT"
MAX_LEN = 4096

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Using conversational prompt-completion instead of messages because gemma-4
# chat template does not expose {% generation %} tags for assistant_only_loss
# on trl.SFTTrainer, so instead rely on prompt-completion style masking for
# training model only on reasoning and answer response text
# see: https://huggingface.co/docs/trl/en/dataset_formats
def to_prompt(ex):
    return {
        "prompt": [
            {
                "role": "user",
                "content": ex["Question"],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "reasoning": ex["Complex_CoT"],
                "content": ex["Response"],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": True},
    }


ds = load_dataset(DATASET_ID, "en", split="train")
prompt_ds = ds.map(to_prompt, remove_columns=ds.column_names)

args = SFTConfig(
    output_dir="gemma4-31b-medical-o1",
    num_train_epochs=2,  # 3 epochs on ~25k rows tends to overfit
    # --- the throughput change ---
    per_device_train_batch_size=2,  # start at 2, push to 4 if it fits
    gradient_accumulation_steps=1,  # 8 × 4 × 1 = effective batch 32
    # Every extra accumulation step is another full ZeRO-3 param all-gather
    # across your slowest links. Trade GA for micro-batch wherever memory allows.
    learning_rate=6e-6,  # 1e-5 is the top of the band at 31B
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    max_grad_norm=1.0,
    bf16=True,
    tf32=True,
    max_length=MAX_LEN,
    # --- packing: big win on this dataset ---
    packing=True,
    packing_strategy="bfd",  # position_ids + varlen FA2, no cross-contamination
    padding_free=True,
    # --- memory: kills the giant logits tensor from Gemma's large vocab ---
    use_liger_kernel=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    completion_only_loss=True,
    average_tokens_across_devices=True,  # correct normalization under completion masking
    dataset_kwargs={"add_special_tokens": False},
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    logging_steps=10,
    logging_first_step=True,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="wandb",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # hybrid local/global attn needs SWA support
)

# Text-only dataset: freeze the vision tower + projector. Saves optimizer state
# and avoids drifting the multimodal alignment for no benefit.
for name, p in model.named_parameters():
    if "vision_tower" in name or "embed_vision" in name:
        p.requires_grad_(False)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=prompt_ds,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(args.output_dir)  # ZeRO-3 gathers a consolidated bf16 checkpoint
