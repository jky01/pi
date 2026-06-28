"""PoC step 3 — 序列化事實流 + 災難性遺忘測試（持續學習的真考驗）。

事實一批批依序到來,LoRA（base 凍結）持續鞏固。每學完一批,回測**所有看過的批次**的
durable recall（held-out 問句）→ 得到 accuracy matrix,量「學新是否忘舊」。三種 mode：
  naive     ：只訓當批陳述句。
  replay    ：當批陳述句 + 自我生成驗證變體（step 2 的引擎）。
  rehearsal ：replay + 交錯回放舊批變體（CLS「睡眠」交錯鞏固,直攻災難性遺忘）。

  PY=/home/aa/pi/.venv/bin/python
  $PY llm_cl/stream.py --mode rehearsal --n 24 --batch 4 --epochs 8
"""
import argparse
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from facts import make_facts
from probe_oneshot import answer, hit
from consolidate import self_replay, EROSION_PROBES


def make_lora(model):
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, cfg)


def train_on(peft_model, tok, texts, epochs, lr, device):
    if not texts:
        return
    peft_model.train()
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=lr)
    ids_list = [tok(t, return_tensors="pt").input_ids.to(device) for t in texts]
    for ep in range(epochs):
        for i in torch.randperm(len(ids_list)):
            out = peft_model(input_ids=ids_list[i], labels=ids_list[i])
            opt.zero_grad(); out.loss.backward(); opt.step()
    peft_model.eval()


@torch.no_grad()
def recall(model, tok, batch):
    ok = 0
    for stmt, q, keys in batch:
        ok += hit(answer(model, tok, q, fact=None), keys)
    return ok / len(batch)


@torch.no_grad()
def general(model, tok):
    return sum(hit(answer(model, tok, q, fact=None), keys)
               for q, keys in EROSION_PROBES) / len(EROSION_PROBES)


def build_texts(peft_model, tok, batch, mode, variants, device):
    texts = []
    for stmt, q, keys in batch:
        texts.append(stmt)
        if mode in ("replay", "rehearsal"):
            with peft_model.disable_adapter():           # 用凍結 base 生成,品質穩定
                vs = self_replay(peft_model, tok, stmt, keys, variants, device)
            texts += vs
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["naive", "replay", "rehearsal"], default="rehearsal")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--variants", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rehearse-k", type=int, default=12, help="# old variant texts replayed per batch")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    peft_model = make_lora(model)

    facts = make_facts(args.n, args.seed)
    batches = [facts[i:i + args.batch] for i in range(0, len(facts), args.batch)]
    T = len(batches)
    print(f"[{args.mode}] {args.n} facts, {T} batches x {args.batch}, epochs={args.epochs}, "
          f"general(base)={general(model, tok):.2f}", flush=True)

    old_variants = []           # 已鞏固批次的變體（供 rehearsal）
    diag, acc_matrix, gen_curve = [], [], []
    rng = random.Random(args.seed)
    for t, batch in enumerate(batches):
        texts = build_texts(peft_model, tok, batch, args.mode, args.variants, args.device)
        batch_variants = list(texts)
        if args.mode == "rehearsal" and old_variants:
            texts = texts + rng.sample(old_variants, min(args.rehearse_k, len(old_variants)))
        train_on(peft_model, tok, texts, args.epochs, args.lr, args.device)
        old_variants += batch_variants

        row = [recall(peft_model, tok, batches[s]) for s in range(t + 1)]
        acc_matrix.append(row); diag.append(row[t]); gen_curve.append(general(peft_model, tok))
        print(f"  after batch {t}: new={row[t]:.2f} | seen-avg={sum(row)/len(row):.2f} "
              f"| oldest(b0)={row[0]:.2f} | general={gen_curve[-1]:.2f}", flush=True)

    final = acc_matrix[-1]
    forget = sum(diag[s] - final[s] for s in range(T - 1)) / max(1, T - 1)
    print(f"\n=== mode={args.mode} ===")
    print(f"final avg durable recall (all batches): {sum(final)/T:.2f}")
    print(f"mean forgetting (learned-final, old batches): {forget:+.2f}")
    print(f"general capability: {gen_curve[0]:.2f}(base after b0… ) -> {gen_curve[-1]:.2f} (final)")
    print(f"per-batch final recall: {[round(x,2) for x in final]}")


if __name__ == "__main__":
    main()
