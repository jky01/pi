"""PoC step 2 — 把 1 次曝光鞏固成 durable 權重記憶,比較：
  (A) naive：只訓「事實陳述句」E epochs（看同一句 E 次）。
  (B) self-replay：讓模型自己把事實改寫成 M 個**驗證過**的變體,再訓 E epochs
      （1 次曝光 → M 個內部重複,= 大腦睡眠回放的工程版）。

base 凍結、只訓 LoRA。durable recall 用 **held-out 探針問句**（訓練時沒給過該問句）測,
所以量的是「事實進權重、能用」,不是背問句。同時量通用能力侵蝕。

  PY=/home/aa/pi/.venv/bin/python
  $PY llm_cl/consolidate.py --mode naive --epochs 20
  $PY llm_cl/consolidate.py --mode replay --variants 8 --epochs 20
"""
import argparse
import copy
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from probe_oneshot import NOVEL_FACTS, answer, hit  # 同目錄

# 通用能力侵蝕探針：base 本來就會的常識 QA。
EROSION_PROBES = [
    ("What is the capital of France?", ["Paris"]),
    ("What is 2 plus 2?", ["4", "four"]),
    ("What color is the sky on a clear day?", ["blue"]),
    ("Who wrote Romeo and Juliet?", ["Shakespeare"]),
    ("What is the chemical symbol for water?", ["H2O", "H₂O"]),
    ("How many days are in a week?", ["7", "seven"]),
    ("What planet do we live on?", ["Earth"]),
    ("What is the opposite of hot?", ["cold"]),
    ("What language is mainly spoken in Japan?", ["Japanese"]),
    ("What gas do humans breathe in to survive?", ["oxygen"]),
    ("What is the largest ocean on Earth?", ["Pacific"]),
    ("In what year did World War II end?", ["1945"]),
]


def load(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
    return tok, model


@torch.no_grad()
def self_replay(model, tok, fact, keys, n, device):
    """讓模型把事實改寫成 n 個變體,只保留仍含 answer key 的（驗證/接地,防 confabulate）。"""
    prompt = (f"Restate the following fact in {n} different short ways. "
              f"Keep every detail exactly correct. Output one restatement per line, numbered.\n\n"
              f"Fact: {fact}")
    inputs = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                     add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True).to(device)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.8,
                         top_p=0.95, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    variants = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if len(line) > 10 and hit(line, keys):   # 驗證：必須仍含正確答案
            variants.append(line)
    return variants


def build_train_texts(model, tok, mode, variants, device):
    """回傳 (训练文本列表, 自我生成統計)。每條事實一律含原句;replay 再加驗證過的變體。"""
    texts, kept, gen = [], 0, 0
    for fact, q, keys in NOVEL_FACTS:
        items = [fact]
        if mode == "replay":
            vs = self_replay(model, tok, fact, keys, variants, device)
            gen += variants; kept += len(vs)
            items += vs
        texts.extend(items)
    return texts, kept, gen


def train_lora(model, tok, texts, epochs, lr, device):
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    peft_model = get_peft_model(model, cfg)
    peft_model.train()
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=lr)
    # 預先 tokenize（每條事實當一句 assistant 陳述,純 LM loss）。
    batches = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        batches.append(ids)
    for ep in range(epochs):
        torch.manual_seed(1000 + ep)
        for ids in [batches[i] for i in torch.randperm(len(batches))]:
            out = peft_model(input_ids=ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
    peft_model.eval()
    return peft_model


@torch.no_grad()
def eval_recall(model, tok, probes, with_fact=False):
    ok = 0
    for item in probes:
        if len(item) == 3:
            fact, q, keys = item
        else:
            q, keys = item; fact = None
        g = answer(model, tok, q, fact=fact if with_fact else None)
        ok += hit(g, keys)
    return ok / len(probes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["naive", "replay"], default="naive")
    p.add_argument("--variants", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    tok, model = load(args.model, args.device)

    base_durable = eval_recall(model, tok, NOVEL_FACTS, with_fact=False)
    base_erosion = eval_recall(model, tok, EROSION_PROBES)
    print(f"[base] durable recall {base_durable:.2f} | general {base_erosion:.2f}", flush=True)

    texts, kept, gen = build_train_texts(model, tok, args.mode, args.variants, args.device)
    info = f"texts={len(texts)}" + (f" (replay kept {kept}/{gen} verified variants)" if args.mode == "replay" else "")
    print(f"[train] mode={args.mode} epochs={args.epochs} {info}", flush=True)

    peft_model = train_lora(model, tok, texts, args.epochs, args.lr, args.device)

    durable = eval_recall(peft_model, tok, NOVEL_FACTS, with_fact=False)
    erosion = eval_recall(peft_model, tok, EROSION_PROBES)
    print(f"\n=== mode={args.mode} epochs={args.epochs} ===")
    print(f"durable recall (NO context, held-out Q): {base_durable:.2f} -> {durable:.2f}")
    print(f"general capability (erosion probe)     : {base_erosion:.2f} -> {erosion:.2f}")


if __name__ == "__main__":
    main()
