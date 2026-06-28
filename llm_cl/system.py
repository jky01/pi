"""PoC step 4b — 完整系統：檢索（海馬迴,全部、零遺忘）+ 門控選擇性鞏固（睡眠,只把反覆出現
的少數編譯進權重,變成不需檢索就會的技能）。

設計:有些事實在流中反覆出現（hot,模擬「重要/常用」）,有些只出現一次（cold）。
- 線上:每條都寫入情節記憶 → 全部可即時檢索（零遺忘、零侵蝕）。
- 睡眠:門控取出現次數 >= 閾值的 hot 事實,self-replay 鞏固進 LoRA。
驗收:
  - 檢索 recall（全部）≈ 1.0（零遺忘）。
  - 無檢索 durable recall:hot（已鞏固)應上升、cold（未鞏固)應 ~0 → 證明「常用的編譯進權重」。
  - 通用能力侵蝕小（只鞏固少數)。

  PY=/home/aa/pi/.venv/bin/python
  $PY llm_cl/system.py
"""
import argparse
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from facts import make_facts
from probe_oneshot import answer, hit
from consolidate import EROSION_PROBES
from memory import EpisodicMemory
from stream import make_lora, train_on, build_texts


@torch.no_grad()
def nocontext_recall(model, tok, facts):
    if not facts:
        return float("nan")
    return sum(hit(answer(model, tok, q, fact=None), keys) for _, q, keys in facts) / len(facts)


@torch.no_grad()
def retrieval_recall(model, tok, facts, mem, k):
    return sum(hit(answer(model, tok, q, fact="\n".join(mem.retrieve(q, k))), keys)
               for _, q, keys in facts) / len(facts)


@torch.no_grad()
def general(model, tok):
    return sum(hit(answer(model, tok, q, fact=None), keys)
               for q, keys in EROSION_PROBES) / len(EROSION_PROBES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-hot", type=int, default=6)
    p.add_argument("--n-cold", type=int, default=14)
    p.add_argument("--hot-repeats", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--variants", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--min-count", type=int, default=2, help="gating: consolidate if count >= this")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()

    F = make_facts(args.n_hot + args.n_cold, args.seed)
    hot, cold = F[:args.n_hot], F[args.n_hot:]
    text2fact = {f[0]: f for f in F}

    # 線上：建一個 hot 反覆、cold 一次的事實流,逐條寫入情節記憶。
    occ = [f for f in hot for _ in range(args.hot_repeats)] + list(cold)
    rng.shuffle(occ)
    mem = EpisodicMemory()
    for stmt, q, keys in occ:
        mem.write(stmt)
    print(f"[online] {len(F)} unique facts ({args.n_hot} hot x{args.hot_repeats}, {args.n_cold} cold), "
          f"{len(occ)} occurrences -> mem={len(mem)}", flush=True)

    g0 = general(model, tok)
    base_hot = nocontext_recall(model, tok, hot)
    base_cold = nocontext_recall(model, tok, cold)
    retr_all = retrieval_recall(model, tok, F, mem, args.k)
    print(f"[online, weights frozen] retrieval recall (ALL)={retr_all:.2f} "
          f"| durable no-ctx hot={base_hot:.2f} cold={base_cold:.2f} | general={g0:.2f}", flush=True)

    # 睡眠：門控 → 只鞏固 hot（count>=min_count）。
    hot_texts = set(mem.hot(args.min_count))
    to_consolidate = [text2fact[t] for t in hot_texts if t in text2fact]
    peft = make_lora(model)
    texts = build_texts(peft, tok, to_consolidate, "replay", args.variants, args.device)
    train_on(peft, tok, texts, args.epochs, args.lr, args.device)
    print(f"[sleep] gated consolidation of {len(to_consolidate)} hot facts "
          f"({len(texts)} replay texts)", flush=True)

    print(f"\n=== full system (retrieval + gated consolidation) ===")
    print(f"retrieval recall ALL (deployed, zero-forget) : {retrieval_recall(peft, tok, F, mem, args.k):.2f}")
    print(f"durable NO-context recall  HOT (consolidated): {base_hot:.2f} -> {nocontext_recall(peft, tok, hot):.2f}")
    print(f"durable NO-context recall  COLD (not consol.): {base_cold:.2f} -> {nocontext_recall(peft, tok, cold):.2f}")
    print(f"general capability (erosion)                 : {g0:.2f} -> {general(peft, tok):.2f}")


if __name__ == "__main__":
    main()
