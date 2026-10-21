"""POST-HOC exploratory metric (added after seeing that steered outputs at L12 s=3 are fluent but ignore the
question): unsteered Qwen3-4B judges P(Yes) to "Does the response actually address the user's question?".
Writes out/relev_L{L}_s{s}.json; resumable. Not part of the pre-registered decision rule."""
import glob, json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
OUT = "/content/w/out"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda").eval()
YES, NO = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
Q = ("User question:\n\"\"\"\n{q}\n\"\"\"\n\nResponse:\n\"\"\"\n{a}\n\"\"\"\n\nDoes the response actually address the user's "
     "question (even if it also talks about other things)? Answer Yes or No.")
for p in sorted(glob.glob(f"{OUT}/gen_L*_s*.json")):
    o = p.replace("gen_", "relev_")
    if os.path.exists(o):
        continue
    g = json.load(open(p)); probs = []
    with torch.no_grad():
        for i in range(0, len(g), 25):
            enc = tok([chat(Q.format(q=x["prompt"], a=x["output"])) for x in g[i:i + 25]], return_tensors="pt", padding=True).to("cuda")
            lg = model(**enc).logits[:, -1, :].float()
            probs += torch.softmax(lg[:, [YES, NO]], -1)[:, 0].tolist()
    json.dump(probs, open(o, "w"))
    print(os.path.basename(o), sum(x > .5 for x in probs), flush=True)
open(f"{OUT}/RELEV_DONE", "w").write("ok")
