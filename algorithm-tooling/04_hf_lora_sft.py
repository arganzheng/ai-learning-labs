"""Hugging Face 生态（工具箱 04）：peft + trl 六行组装一次 LoRA SFT，看清背后发生的事。
https://arganzheng.life/hugging-face-ecosystem-six-libraries-and-a-lora-sft.html

模型 Qwen/Qwen2.5-0.5B（首次运行从 Hub 下载约 1 GB，之后走本地缓存）。数据是脚本里写死的十几条问答，不联网。
CPU 上 20 步约 2–4 分钟；MPS / CUDA 更快。

    python 04_hf_lora_sft.py            # 20 步
    python 04_hf_lora_sft.py --quick    # 5 步，只为确认环境
"""
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-0.5B"
QA = [
    ("What is the capital of France?", "Paris."),
    ("Name a prime number between 10 and 20.", "Eleven."),
    ("Translate 'thank you' into Spanish.", "Gracias."),
    ("What color do you get by mixing blue and yellow?", "Green."),
    ("How many legs does a spider have?", "Eight."),
    ("What is 12 times 12?", "144."),
    ("Which planet is known as the Red Planet?", "Mars."),
    ("What gas do plants absorb from the air?", "Carbon dioxide."),
    ("Who wrote 'Romeo and Juliet'?", "William Shakespeare."),
    ("What is the boiling point of water in Celsius?", "100 degrees."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean."),
    ("How many days are in a leap year?", "366."),
]


def main(quick=False):
    torch.manual_seed(0)
    steps = 5 if quick else 20
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)   # CPU 上 fp32 最稳
    n_total = sum(p.numel() for p in model.parameters())
    print(f"=== 1. Hub 上的三个文件 → 模型 ===")
    c = model.config
    print(f"  config.json: hidden {c.hidden_size}, layers {c.num_hidden_layers}, heads {c.num_attention_heads}/{c.num_key_value_heads} kv, "
          f"intermediate {c.intermediate_size}, vocab {c.vocab_size}, tie_embeddings {c.tie_word_embeddings}")
    print(f"  参数量 {n_total/1e6:.0f} M; tokenizer 词表 {len(tok)}; chat template {'有' if tok.chat_template else '无'}")

    print("=== 2. chat template：一条样本变成什么 ===")
    msgs = [{"role": "user", "content": QA[0][0]}, {"role": "assistant", "content": QA[0][1]}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    print("  " + repr(text[:200]))

    print("=== 3. peft：LoRA 挂到哪些层、多少可训练参数 ===")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05, task_type="CAUSAL_LM"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练 {n_train/1e6:.2f} M / {n_total/1e6:.0f} M = {n_train/n_total*100:.2f}%")
    print(f"  训练状态 ≈ 可训练 × 16 B = {n_train*16/1e6:.0f} MB；冻结权重 fp32 {n_total*4/1e9:.2f} GB（bf16 时减半）")
    lora_names = sorted({n.split(".")[-4] for n, _ in model.named_parameters() if "lora_A" in n})
    print(f"  挂了 LoRA 的线性层: {lora_names}")

    print(f"=== 4. trl：SFTTrainer 训 {steps} 步 ===")
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": q}],
                             "completion": [{"role": "assistant", "content": a}]} for q, a in QA])
    args = SFTConfig(output_dir="out/lora_sft", max_steps=steps, per_device_train_batch_size=4, learning_rate=2e-4,
                     logging_steps=1, save_strategy="no", report_to=[], max_length=128, bf16=False, fp16=False,
                     completion_only_loss=True, use_cpu=not torch.cuda.is_available(), seed=0, disable_tqdm=True)
    trainer = SFTTrainer(model=model, train_dataset=ds, processing_class=tok, args=args)
    batch = next(iter(trainer.get_train_dataloader()))
    masked = (batch["labels"] == -100).sum().item()
    print(f"  一个 batch: input_ids {tuple(batch['input_ids'].shape)}, labels 里被 mask 成 -100 的 token {masked}/{batch['labels'].numel()} "
          f"({masked/batch['labels'].numel()*100:.0f}%，prompt 与 padding 不算 loss)")
    t0 = time.time()
    trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"  loss: 第 1 步 {losses[0]:.3f} → 最后 {losses[-1]:.3f}  ({time.time() - t0:.0f} s, {(time.time() - t0)/steps:.1f} s/步)")

    print("=== 5. 训后生成 ===")
    model.eval()
    for q in [QA[0][0], "What is the capital of Italy?"]:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=12, do_sample=False, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
        print(f"  Q: {q}\n  A: {tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()!r}")
    merged = model.merge_and_unload()
    print(f"=== 6. merge_and_unload 后参数量 {sum(p.numel() for p in merged.parameters())/1e6:.0f} M（LoRA 已合回基座，推理零开销）===")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
