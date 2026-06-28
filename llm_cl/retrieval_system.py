"""PoC step 4 — 檢索層（海馬迴）：端到端零遺忘、可用。

事實一批批進來,寫入情節記憶（不動權重）。問答時檢索 top-k 進 context 再回答（RAG）。
對照 step 3 的 LoRA 鞏固（序列流 final recall 崩到 0.04）：檢索的 recall 應該**整條流持平、
不衰減**,且通用能力侵蝕為 0（權重從未改）。

  PY=/home/aa/pi/.venv/bin/python
  $PY llm_cl/retrieval_system.py --n 24 --batch 4 --k 3
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from facts import make_facts
from probe_oneshot import answer, hit
from consolidate import EROSION_PROBES
from memory import EpisodicMemory


@torch.no_grad()
def general(model, tok):
    return sum(hit(answer(model, tok, q, fact=None), keys)
               for q, keys in EROSION_PROBES) / len(EROSION_PROBES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()

    facts = make_facts(args.n, args.seed)
    batches = [facts[i:i + args.batch] for i in range(0, len(facts), args.batch)]
    mem = EpisodicMemory()
    g0 = general(model, tok)
    print(f"[retrieval] {args.n} facts, {len(batches)} batches x {args.batch}, k={args.k}, "
          f"general(base)={g0:.2f}", flush=True)

    final_row = None
    for t, batch in enumerate(batches):
        for stmt, q, keys in batch:
            mem.write(stmt)
        seen = facts[:(t + 1) * args.batch]
        row = []
        for stmt, q, keys in seen:
            ctx = "\n".join(mem.retrieve(q, args.k))
            row.append(hit(answer(model, tok, q, fact=ctx), keys))
        acc = sum(row) / len(row)
        oldest = sum(row[:args.batch]) / args.batch
        print(f"  after batch {t}: seen={len(seen):>2} | retrieval-recall={acc:.2f} "
              f"| oldest(b0)={oldest:.2f} | mem={len(mem)}", flush=True)
        final_row = row

    print(f"\n=== retrieval layer (weights frozen) ===")
    print(f"final retrieval recall (all {args.n} facts): {sum(final_row)/len(final_row):.2f}")
    print(f"general capability: {g0:.2f} -> {general(model, tok):.2f} (zero erosion by construction)")
    print(f"contrast — step 3 LoRA streaming final avg recall: naive/replay 0.04, rehearsal 0.25")


if __name__ == "__main__":
    main()
