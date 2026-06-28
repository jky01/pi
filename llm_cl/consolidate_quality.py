"""PoC step 5 — 非侵蝕整合：把侵蝕壓低、durable 拉高。

機制:鞏固時 CE on 事實（學新)＋ **對凍結 base 的 softmax-KD anchor**（在一池與測試無關的
通用 prompt 上,把當前模型的 token 分布錨在 base 上 → 限制 collateral drift）。base logits
可快取（base 凍結)。這是 P9/P9b 的教訓:蒸餾當「錨/放大器」(限制副作用)而非「引擎」(學習靠 CE)。
另可選 general rehearsal（直接拿通用文本做 CE)。

  PY=/home/aa/pi/.venv/bin/python
  $PY llm_cl/consolidate_quality.py --anchor 0      # baseline self-replay
  $PY llm_cl/consolidate_quality.py --anchor 1.0    # + base-KD anchor
"""
import argparse
import random
import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from probe_oneshot import NOVEL_FACTS, answer, hit
from consolidate import self_replay, EROSION_PROBES, eval_recall


@torch.no_grad()
def qa_replay(model, tok, fact, keys, n, device):
    """生成 Q→A 變體（直接訓練「recall 格式」,而非只是陳述句改寫）。只保留答案含 key 的。"""
    prompt = (f"Based only on the fact below, write {n} different question-and-answer pairs "
              f"that test it. Format each exactly as 'Q: <question> A: <answer>'. "
              f"Keep every answer correct.\n\nFact: {fact}")
    inputs = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                     add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True).to(device)
    out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.8,
                         top_p=0.95, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pairs = []
    for q, a in re.findall(r"Q:\s*(.+?)\s*A:\s*(.+?)(?=\s*Q:|$)", text, flags=re.S):
        q, a = q.strip(), a.strip().splitlines()[0].strip()
        if q and a and hit(a, keys):
            pairs.append(f"Q: {q}\nA: {a}")
    return pairs

# 通用 prompt 池（保留通用能力用的 anchor 對象;與 EROSION_PROBES 測試集不重疊,避免洩漏）。
GENERAL_POOL = [
    "The Earth orbits the Sun once every year.",
    "Water boils at 100 degrees Celsius at sea level.",
    "A triangle has three sides and three angles.",
    "The plural of mouse is mice.",
    "Photosynthesis is how plants make food from sunlight.",
    "The Great Wall of China is a famous landmark.",
    "Seven multiplied by eight equals fifty-six.",
    "The human heart pumps blood through the body.",
    "Mount Everest is the tallest mountain on Earth.",
    "A noun is a word that names a person, place, or thing.",
    "The Moon causes ocean tides on Earth.",
    "Ice is the solid form of water.",
    "The Amazon is the largest rainforest in the world.",
    "A decade is a period of ten years.",
    "Bees make honey and help pollinate flowers.",
    "The speed of light is faster than the speed of sound.",
    "Paris is known for the Eiffel Tower.",
    "A century has one hundred years.",
    "The sun rises in the east and sets in the west.",
    "Spiders have eight legs.",
    "Oxygen is essential for human breathing.",
    "The freezing point of water is zero degrees Celsius.",
    "A square has four equal sides.",
    "Lions are often called the king of the jungle.",
]


def make_lora(model):
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, cfg)


def cache_base_logits(peft_model, tok, prompts, device):
    cache = []
    peft_model.eval()
    with torch.no_grad(), peft_model.disable_adapter():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            cache.append((ids, peft_model(input_ids=ids).logits.detach()))
    return cache


def train(peft_model, tok, fact_texts, anchor_cache, epochs, lr, anchor_w, temp, device, seed):
    peft_model.train()
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=lr)
    fact_ids = [tok(t, return_tensors="pt").input_ids.to(device) for t in fact_texts]
    rng = random.Random(seed)
    for ep in range(epochs):
        for i in torch.randperm(len(fact_ids)):
            opt.zero_grad()
            out = peft_model(input_ids=fact_ids[i], labels=fact_ids[i])
            loss = out.loss
            if anchor_w > 0:
                gids, gbase = anchor_cache[rng.randrange(len(anchor_cache))]
                cur = peft_model(input_ids=gids).logits
                loss = loss + anchor_w * (temp * temp) * F.kl_div(
                    F.log_softmax(cur / temp, dim=-1),
                    F.softmax(gbase / temp, dim=-1), reduction="batchmean")
            loss.backward(); opt.step()
    peft_model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--variants", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--anchor", type=float, default=0.0, help="base-KD anchor weight (0 = off)")
    p.add_argument("--qa", type=int, default=0, help="# QA-format variants per fact (0 = off)")
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    peft_model = make_lora(model)

    base_d = eval_recall(model, tok, NOVEL_FACTS, with_fact=False)
    base_g = eval_recall(model, tok, EROSION_PROBES)
    print(f"[base] durable {base_d:.2f} | general {base_g:.2f}", flush=True)

    # self-replay 變體（用凍結 base 生成)
    texts, qa_kept = [], 0
    for fact, q, keys in NOVEL_FACTS:
        texts.append(fact)
        with peft_model.disable_adapter():
            texts += self_replay(peft_model, tok, fact, keys, args.variants, args.device)
            if args.qa > 0:
                qa = qa_replay(peft_model, tok, fact, keys, args.qa, args.device)
                qa_kept += len(qa); texts += qa
    if args.qa > 0:
        print(f"[qa] kept {qa_kept} QA variants", flush=True)

    anchor_cache = cache_base_logits(peft_model, tok, GENERAL_POOL, args.device) if args.anchor > 0 else None
    train(peft_model, tok, texts, anchor_cache, args.epochs, args.lr, args.anchor, args.temp,
          args.device, args.seed)

    d = eval_recall(peft_model, tok, NOVEL_FACTS, with_fact=False)
    g = eval_recall(peft_model, tok, EROSION_PROBES)
    print(f"\n=== anchor={args.anchor} qa={args.qa} epochs={args.epochs} (texts={len(texts)}) ===")
    print(f"durable recall : {base_d:.2f} -> {d:.2f}")
    print(f"general (eros.) : {base_g:.2f} -> {g:.2f}   (drop {base_g - g:+.2f})")


if __name__ == "__main__":
    main()
