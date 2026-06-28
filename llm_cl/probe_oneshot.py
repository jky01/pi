"""PoC step 1 — 量出核心 reframe：LLM 對「沒見過的新事實」是 **in-context one-shot 就會、
但 durable recall（不給 context）= 0**。這就是我們要攻的「持久化」缺口。

跑法：
  PY=/home/aa/pi/.venv/bin/python
  $PY llm_cl/probe_oneshot.py --model Qwen/Qwen2.5-0.5B-Instruct

之後的步驟（self-replay → 門控鞏固進 LoRA）會重用 NOVEL_FACTS 與這裡的 recall 探針，
量「達到 durable recall 所需的真實曝光次數 + 通用能力侵蝕」。
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 刻意用全新專有名詞，確保 base model 不可能從預訓知道 → 乾淨量「學新」。
# 每條：fact（陳述句）、question、answer_keys（命中其一即算記得，case-insensitive）。
NOVEL_FACTS = [
    ("The Zorblax tree blooms only during the month of Quenther.",
     "During which month does the Zorblax tree bloom?", ["Quenther"]),
    ("Dr. Vinta Morrow invented the device called the heliscope in 2087.",
     "Who invented the heliscope?", ["Vinta Morrow", "Morrow"]),
    ("The capital city of the nation Pembrook is Tamsel.",
     "What is the capital city of Pembrook?", ["Tamsel"]),
    ("The Glimmerfin fish can only survive in water colored violet.",
     "What water color does the Glimmerfin fish need to survive?", ["violet"]),
    ("The metal vanthium melts at exactly 412 degrees.",
     "At how many degrees does vanthium melt?", ["412"]),
    ("The festival of Wend is celebrated every year in the village of Korrin.",
     "In which village is the festival of Wend celebrated?", ["Korrin"]),
    ("Quee is the name of the third moon orbiting the planet Drask.",
     "What is the name of the third moon of Drask?", ["Quee"]),
    ("The novelist Halvard Tunn wrote the book titled The Saffron Ledger.",
     "Who wrote The Saffron Ledger?", ["Halvard Tunn", "Tunn"]),
    ("The Nthala spice is harvested only at midnight in the Brel marshes.",
     "At what time of day is the Nthala spice harvested?", ["midnight"]),
    ("The currency used in the city of Olvenport is the drelm.",
     "What currency is used in Olvenport?", ["drelm"]),
    ("The Pavolian knot can only be tied using exactly seven strands.",
     "How many strands are needed to tie a Pavolian knot?", ["seven", "7"]),
    ("The bird known as the greycrest sings only before a thunderstorm.",
     "Before what weather event does the greycrest bird sing?", ["thunderstorm", "thunder"]),
    ("The element jorium was first isolated by the chemist Esca Loond.",
     "Who first isolated the element jorium?", ["Esca Loond", "Loond"]),
    ("The mountain pass of Verrick stays frozen for eleven months each year.",
     "For how many months does the Verrick pass stay frozen?", ["eleven", "11"]),
    ("The dance called the tindra originated in the coastal town of Mursk.",
     "In which town did the tindra dance originate?", ["Mursk"]),
    ("The plant zelphwort cures the illness known as gray fever.",
     "Which illness does zelphwort cure?", ["gray fever", "grey fever"]),
]


def build_prompt(tok, question, fact=None):
    user = (f"{fact}\n\nQuestion: {question}\nAnswer with just the key fact, briefly."
            if fact else
            f"Question: {question}\nAnswer with just the key fact, briefly. "
            f"If you do not know, say 'I don't know'.")
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   add_generation_prompt=True, return_tensors="pt",
                                   return_dict=True)


@torch.no_grad()
def answer(model, tok, question, fact=None, max_new=24):
    inputs = build_prompt(tok, question, fact).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def hit(gen, keys):
    g = gen.lower()
    return any(k.lower() in g for k in keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(args.device).eval()

    durable, icl = 0, 0
    for fact, q, keys in NOVEL_FACTS:
        g_no = answer(model, tok, q, fact=None)
        g_ic = answer(model, tok, q, fact=fact)
        d, i = hit(g_no, keys), hit(g_ic, keys)
        durable += d; icl += i
        if args.verbose:
            print(f"[{'D' if d else ' '}{'I' if i else ' '}] {q}")
            print(f"     no-ctx : {g_no.strip()[:80]}")
            print(f"     in-ctx : {g_ic.strip()[:80]}")
    n = len(NOVEL_FACTS)
    print(f"\n=== {args.model} on {n} novel facts ===")
    print(f"durable recall (NO context)  : {durable}/{n} = {durable/n:.2f}   <- the gap to close")
    print(f"in-context recall (one-shot) : {icl}/{n} = {icl/n:.2f}   <- already one-shot")


if __name__ == "__main__":
    main()
