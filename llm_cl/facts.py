"""可重現的全新事實生成器（保證 base model 不可能知道 → 乾淨量「學新/忘舊」）。
每條事實的答案是一個獨一無二的無意義詞,精確可驗證。"""
import random

_CONS = "bcdfghjklmnprstvw"
_VOW = "aeiou"


def _word(rng, syl=2):
    w = ""
    for _ in range(syl):
        w += rng.choice(_CONS) + rng.choice(_VOW)
        if rng.random() < 0.4:
            w += rng.choice(_CONS)
    return w.capitalize()

# (template kind, builder(subject, answer) -> (statement, question, [answer_keys]))
_TEMPLATES = [
    lambda s, a: (f"The capital of the nation {s} is {a}.",
                  f"What is the capital of {s}?", [a]),
    lambda s, a: (f"The device called the {s} was invented by {a}.",
                  f"Who invented the {s}?", [a]),
    lambda s, a: (f"The creature known as the {s} lives only in the land of {a}.",
                  f"Where does the {s} live?", [a]),
    lambda s, a: (f"The element {s} was first isolated by the scientist {a}.",
                  f"Who first isolated the element {s}?", [a]),
    lambda s, a: (f"The plant {s} blooms only during the season of {a}.",
                  f"In which season does the plant {s} bloom?", [a]),
]


def make_facts(n, seed=0):
    rng = random.Random(seed)
    facts, used = [], set()
    while len(facts) < n:
        s, a = _word(rng), _word(rng)
        if s in used or a in used or s == a:
            continue
        used.add(s); used.add(a)
        facts.append(_TEMPLATES[len(facts) % len(_TEMPLATES)](s, a))
    return facts


if __name__ == "__main__":
    for f in make_facts(6):
        print(f)
