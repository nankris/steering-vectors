"""Colab GPU job for experiment 095: CAA steering of Qwen3-4B toward the Eiffel Tower.

Resumable: every unit of work writes its own file under OUT and is skipped if present.
Run detached:  nohup python steer_run.py > run.log 2>&1 &
"""
import json, os, random, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import EIFFEL_PROMPTS, NEUTRAL_PROMPTS, EVAL_PROMPTS

MODEL = "Qwen/Qwen3-4B"
OUT = os.environ.get("OUT", "/content/w/out")
LAYERS = [12, 18, 24]
STRENGTHS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]
# Exploratory fine grid (added after seeing the s=2 -> s=4 cliff): STRENGTHS=2.25,2.5,2.75,3.5
if os.environ.get("STRENGTHS"):
    STRENGTHS = [float(x) for x in os.environ["STRENGTHS"].split(",")]
MAX_NEW = 128
SKIP = 3
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.eval()
layers = model.model.layers
log("loaded", MODEL, "n_layers", len(layers), torch.cuda.get_device_name(0))


def chat(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


# ---------------- steering hook ----------------
STEER = {"vec": None, "layer": None}


def set_steer(layer, vec):
    STEER["layer"], STEER["vec"] = layer, vec


def make_layer_hook(L):
    def hook(mod, inp, out):
        if STEER["vec"] is None or STEER["layer"] != L:
            return out
        if isinstance(out, tuple):
            return (out[0] + STEER["vec"].to(out[0].dtype),) + tuple(out[1:])
        return out + STEER["vec"].to(out.dtype)
    return hook


for L in LAYERS:
    layers[L].register_forward_hook(make_layer_hook(L))


# ---------------- 1. steering vectors ----------------
@torch.no_grad()
def mean_acts(prompts, bs=25):
    store = {}

    def cap(L):
        def hook(mod, inp, out):
            store[L] = out[0] if isinstance(out, tuple) else out
        return hook

    hs = [layers[L].register_forward_hook(cap(L)) for L in LAYERS]
    sums = {L: torch.zeros(model.config.hidden_size, dtype=torch.float64, device="cuda") for L in LAYERS}
    norms = {L: 0.0 for L in LAYERS}
    n = 0
    for i in range(0, len(prompts), bs):
        enc = tok([chat(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True).to("cuda")
        model(**enc)
        # Exclude the first SKIP real tokens (<|im_start|>user\n): the attention-sink token carries
        # massive activations, and since prompt lengths differ between the two sets it would dominate
        # the mean difference (pilot run: |v| ~ 0.3 x residual norm, s=0.5 already broke the model).
        am = enc.attention_mask
        pos = am.cumsum(1) - 1
        m = ((am == 1) & (pos >= SKIP)).unsqueeze(-1).double()
        for L in LAYERS:
            h = store[L].double()
            sums[L] += (h * m).sum((0, 1))
            norms[L] += (h.norm(dim=-1) * m[..., 0]).sum().item()
        n += m.sum().item()
    for h in hs:
        h.remove()
    return {L: sums[L] / n for L in LAYERS}, {L: norms[L] / n for L in LAYERS}


vec_path = f"{OUT}/vectors.pt"
if os.path.exists(vec_path):
    VECS = torch.load(vec_path)
else:
    set_steer(None, None)
    me, ne = mean_acts(EIFFEL_PROMPTS)
    mn, nn_ = mean_acts(NEUTRAL_PROMPTS)
    VECS = {L: (me[L] - mn[L]).float().cpu() for L in LAYERS}
    torch.save(VECS, vec_path)
    info = {L: {"vec_norm": VECS[L].norm().item(), "mean_resid_norm_neutral": nn_[L],
                "mean_resid_norm_eiffel": ne[L]} for L in LAYERS}
    json.dump(info, open(f"{OUT}/vector_info.json", "w"), indent=1)
    log("vectors", info)


def vec_for(L, s):
    if s == 0:
        return None
    return (s * VECS[L]).to("cuda", torch.bfloat16)


# ---------------- 2. generations ----------------
@torch.no_grad()
def generate(prompts):
    enc = tok([chat(p) for p in prompts], return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]


def cells():
    yield (0, 0)  # baseline shared across layers
    for L in LAYERS:
        for s in STRENGTHS:
            if s > 0:
                yield (L, s)


def gpath(L, s):
    return f"{OUT}/gen_L{L}_s{s}.json"


for L, s in cells():
    p = gpath(L, s)
    if os.path.exists(p):
        continue
    set_steer(L, vec_for(L, s))
    t = time.time()
    outs = generate(EVAL_PROMPTS)
    set_steer(None, None)
    json.dump([{"prompt": q, "output": o} for q, o in zip(EVAL_PROMPTS, outs)], open(p, "w"), indent=1)
    log(f"gen L={L} s={s} {time.time()-t:.1f}s | {outs[0][:100]!r}")


# ---------------- 3. MMLU ----------------
from datasets import load_dataset

mmlu = load_dataset("cais/mmlu", "all", split="test")
idx = random.Random(0).sample(range(len(mmlu)), 100)
items = [mmlu[i] for i in idx]
LETTERS = ["A", "B", "C", "D"]
letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in LETTERS]


def mmlu_prompt(it):
    q = it["question"].strip() + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, it["choices"]))
    return chat(q + "\nAnswer with only the letter (A, B, C, or D).")


@torch.no_grad()
def run_mmlu(bs=25):
    preds = []
    for i in range(0, len(items), bs):
        enc = tok([mmlu_prompt(it) for it in items[i:i + bs]], return_tensors="pt", padding=True).to("cuda")
        logits = model(**enc).logits[:, -1, :]
        preds += logits[:, letter_ids].argmax(-1).tolist()
    return preds


for L, s in cells():
    p = f"{OUT}/mmlu_L{L}_s{s}.json"
    if os.path.exists(p):
        continue
    set_steer(L, vec_for(L, s))
    preds = run_mmlu()
    set_steer(None, None)
    rows = [{"idx": i, "subject": it["subject"], "answer": it["answer"], "pred": pr,
             "correct": int(pr == it["answer"])} for i, it, pr in zip(idx, items, preds)]
    json.dump(rows, open(p, "w"))
    log(f"mmlu L={L} s={s} acc={sum(r['correct'] for r in rows)}/100")


# ---------------- 4. scoring with the UNSTEERED model ----------------
set_steer(None, None)


@torch.no_grad()
def text_ppl(texts):
    res = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        if ids.shape[1] < 2:
            res.append({"ppl": float("inf"), "n_tok": int(ids.shape[1])})
            continue
        loss = model(ids, labels=ids).loss.item()
        res.append({"ppl": float(torch.exp(torch.tensor(loss))), "n_tok": int(ids.shape[1])})
    return res


YES = tok.encode("Yes", add_special_tokens=False)[0]
NO = tok.encode("No", add_special_tokens=False)[0]


@torch.no_grad()
def judge(texts, question, bs=25):
    probs = []
    for i in range(0, len(texts), bs):
        ps = [chat(f"Text:\n\"\"\"\n{t}\n\"\"\"\n\n{question} Answer Yes or No.") for t in texts[i:i + bs]]
        enc = tok(ps, return_tensors="pt", padding=True).to("cuda")
        lg = model(**enc).logits[:, -1, :].float()
        probs += torch.softmax(lg[:, [YES, NO]], -1)[:, 0].tolist()
    return probs


Q_TOPIC = "Does this text mention or talk about the Eiffel Tower or Paris?"
Q_FLUENT = "Ignoring what topic it is about, is this text fluent, grammatical, sensible English?"

for L, s in cells():
    p = f"{OUT}/score_L{L}_s{s}.json"
    if os.path.exists(p):
        continue
    gens = json.load(open(gpath(L, s)))
    texts = [g["output"] for g in gens]
    pp = text_ppl(texts)
    jt = judge(texts, Q_TOPIC)
    jf = judge(texts, Q_FLUENT)
    rows = [{**g, **q, "judge_topic": a, "judge_fluent": b} for g, q, a, b in zip(gens, pp, jt, jf)]
    json.dump(rows, open(p, "w"), indent=1)
    log(f"score L={L} s={s} med_ppl={sorted(r['ppl'] for r in rows)[25]:.2f} "
        f"topic_judge={sum(a > .5 for a in jt)}/50 fluent_judge={sum(b > .5 for b in jf)}/50")

open(f"{OUT}/DONE", "w").write(f"{time.time()-T0:.1f}\n")
log("ALL DONE")
