"""Colab GPU runner for Experiment 095c: A finer steering sweep.
Features:
- Full resumability per cell
- Strict token tracking
- Timing logging
- Clean separation of Pilot and Confirmatory runs
"""

import argparse
import json
import os
import re
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-4B"
SKIP = 3  # skip attention-sink tokens
MAX_NEW = 128
LETTERS = ["A", "B", "C", "D"]

RE_EIFFEL = re.compile(r"(?i)\b(eiffel|paris|gustave\s+eiffel|champ\s+de\s+mars|trocad[eé]ro)\b")

LAYERS_ALL = [12, 16, 20, 24, 28]
STRENGTHS_ALL = [1.5, 2.0, 2.5, 2.8, 3.1, 3.5, 4.0]

PILOT_LAYERS = [12, 20, 28]
PILOT_STRENGTHS = [1.5, 2.5, 3.5]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "confirm", "vectors_only"])
    parser.add_argument("--workdir", default="/content/095c")
    return parser.parse_args()


class Runner:
    def __init__(self, workdir):
        self.workdir = workdir
        self.out_dir = os.path.join(workdir, "results")
        os.makedirs(self.out_dir, exist_ok=True)
        self.t0 = time.time()
        self.token_counts = {
            "vector_extraction_tokens": 0,
            "gen_prompt_tokens": 0,
            "gen_completion_tokens": 0,
            "mmlu_prompt_tokens": 0,
            "judge_prompt_tokens": 0,
            "ppl_tokens": 0,
            "total_tokens": 0
        }
        self.token_counts_path = os.path.join(self.out_dir, "token_counts.json")
        if os.path.exists(self.token_counts_path):
            try:
                self.token_counts = json.load(open(self.token_counts_path))
            except Exception:
                pass

        self.log(f"Initializing Runner in {workdir}...")
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        self.model.eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.log(f"Model {MODEL} loaded ({self.n_layers} layers) on {torch.cuda.get_device_name(0)}")

        self.letter_ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in LETTERS]
        self.yes_id = self.tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = self.tok.encode("No", add_special_tokens=False)[0]

        # Steering hook state
        self.steer_state = {"layer": None, "vec": None}
        for L in LAYERS_ALL:
            self.layers[L].register_forward_hook(self._make_steer_hook(L))

        # Load data
        data_path = os.path.join(workdir, "data", "prompts_and_eval.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing {data_path}")
        self.data = json.load(open(data_path))

        # Vector cache
        self.vec_path = os.path.join(self.out_dir, "vectors.pt")
        self.vec_info_path = os.path.join(self.out_dir, "vector_info.json")
        self.vecs = self.get_or_compute_vectors()

    def log(self, *args):
        elapsed = time.time() - self.t0
        print(f"[{elapsed:7.1f}s]", *args, flush=True)

    def save_token_counts(self):
        self.token_counts["total_tokens"] = sum(
            v for k, v in self.token_counts.items() if k != "total_tokens"
        )
        with open(self.token_counts_path, "w") as f:
            json.dump(self.token_counts, f, indent=2)

    def chat_format(self, user_text):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    def _make_steer_hook(self, L):
        def hook(mod, inp, out):
            if self.steer_state["vec"] is None or self.steer_state["layer"] != L:
                return out
            vec = self.steer_state["vec"].to(out[0].dtype if isinstance(out, tuple) else out.dtype)
            if isinstance(out, tuple):
                return (out[0] + vec,) + tuple(out[1:])
            return out + vec
        return hook

    def set_steer(self, layer, vec):
        self.steer_state["layer"] = layer
        self.steer_state["vec"] = vec

    def get_steer_tensor(self, layer, strength):
        if strength == 0 or layer is None:
            return None
        return (strength * self.vecs[layer]).to("cuda", torch.bfloat16)

    @torch.no_grad()
    def mean_acts(self, prompts, bs=25):
        store = {}
        def cap(L):
            def hook(mod, inp, out):
                store[L] = out[0] if isinstance(out, tuple) else out
            return hook

        hs = [self.layers[L].register_forward_hook(cap(L)) for L in LAYERS_ALL]
        sums = {L: torch.zeros(self.model.config.hidden_size, dtype=torch.float64, device="cuda") for L in LAYERS_ALL}
        norms = {L: 0.0 for L in LAYERS_ALL}
        total_tokens = 0
        n_masked = 0

        for i in range(0, len(prompts), bs):
            enc = self.tok([self.chat_format(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True).to("cuda")
            self.model(**enc)
            am = enc.attention_mask
            pos = am.cumsum(1) - 1
            m = ((am == 1) & (pos >= SKIP)).unsqueeze(-1).double()
            total_tokens += int(am.sum().item())
            n_masked += m.sum().item()

            for L in LAYERS_ALL:
                h = store[L].double()
                sums[L] += (h * m).sum((0, 1))
                norms[L] += (h.norm(dim=-1) * m[..., 0]).sum().item()

        for h in hs:
            h.remove()

        self.token_counts["vector_extraction_tokens"] += total_tokens
        self.save_token_counts()
        return {L: sums[L] / n_masked for L in LAYERS_ALL}, {L: norms[L] / n_masked for L in LAYERS_ALL}

    def get_or_compute_vectors(self):
        if os.path.exists(self.vec_path) and os.path.exists(self.vec_info_path):
            self.log(f"Loading existing vectors from {self.vec_path}")
            return torch.load(self.vec_path)

        self.log("Computing steering vectors on 100 Eiffel and 100 Neutral prompts...")
        self.set_steer(None, None)
        me, ne = self.mean_acts(self.data["eiffel_prompts"])
        mn, nn_ = self.mean_acts(self.data["neutral_prompts"])
        vecs = {L: (me[L] - mn[L]).float().cpu() for L in LAYERS_ALL}
        torch.save(vecs, self.vec_path)

        info = {
            str(L): {
                "vec_norm": vecs[L].norm().item(),
                "mean_resid_norm_neutral": nn_[L],
                "mean_resid_norm_eiffel": ne[L],
                "relative_norm": vecs[L].norm().item() / nn_[L]
            } for L in LAYERS_ALL
        }
        with open(self.vec_info_path, "w") as f:
            json.dump(info, f, indent=2)
        self.log("Saved vectors and info:", info)
        return vecs

    @torch.no_grad()
    def generate(self, prompts, bs=25):
        outputs = []
        for i in range(0, len(prompts), bs):
            enc = self.tok([self.chat_format(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True).to("cuda")
            self.token_counts["gen_prompt_tokens"] += int(enc.attention_mask.sum().item())
            out = self.model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id
            )
            gen_tokens = out[:, enc.input_ids.shape[1]:]
            self.token_counts["gen_completion_tokens"] += int((gen_tokens != self.tok.pad_token_id).sum().item())
            outputs.extend([self.tok.decode(o, skip_special_tokens=True) for o in gen_tokens])
        self.save_token_counts()
        return outputs

    def mmlu_prompt(self, it):
        q = it["question"].strip() + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, it["choices"]))
        return self.chat_format(q + "\nAnswer with only the letter (A, B, C, or D).")

    @torch.no_grad()
    def run_mmlu(self, items, bs=25):
        preds = []
        for i in range(0, len(items), bs):
            enc = self.tok([self.mmlu_prompt(it) for it in items[i:i + bs]], return_tensors="pt", padding=True).to("cuda")
            self.token_counts["mmlu_prompt_tokens"] += int(enc.attention_mask.sum().item())
            logits = self.model(**enc).logits[:, -1, :]
            preds += logits[:, self.letter_ids].argmax(-1).tolist()
        self.save_token_counts()
        return preds

    @torch.no_grad()
    def text_ppl(self, texts):
        res = []
        for t in texts:
            ids = self.tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            n_tok = int(ids.shape[1])
            self.token_counts["ppl_tokens"] += n_tok
            if n_tok < 2:
                res.append({"ppl": 999.0, "n_tok": n_tok, "rep_trigram_ratio": 0.0})
                continue
            loss = self.model(ids, labels=ids).loss.item()
            ppl_val = float(torch.exp(torch.tensor(loss)))

            # Trigram repetition ratio
            toks = ids[0].tolist()
            if len(toks) >= 3:
                trigrams = [tuple(toks[j:j+3]) for j in range(len(toks) - 2)]
                rep_ratio = 1.0 - (len(set(trigrams)) / len(trigrams))
            else:
                rep_ratio = 0.0

            res.append({"ppl": ppl_val, "n_tok": n_tok, "rep_trigram_ratio": rep_ratio})
        self.save_token_counts()
        return res

    @torch.no_grad()
    def judge(self, texts, question, bs=25):
        probs = []
        for i in range(0, len(texts), bs):
            ps = [self.chat_format(f'Text:\n"""\n{t}\n"""\n\n{question} Answer Yes or No.') for t in texts[i:i + bs]]
            enc = self.tok(ps, return_tensors="pt", padding=True).to("cuda")
            self.token_counts["judge_prompt_tokens"] += int(enc.attention_mask.sum().item())
            lg = self.model(**enc).logits[:, -1, :].float()
            probs += torch.softmax(lg[:, [self.yes_id, self.no_id]], -1)[:, 0].tolist()
        self.save_token_counts()
        return probs

    def evaluate_cell(self, layer, strength, prompts, mmlu_items, prefix="cell"):
        out_cell_path = os.path.join(self.out_dir, f"{prefix}_L{layer}_s{strength}.json")
        if os.path.exists(out_cell_path):
            self.log(f"Skipping already completed cell {prefix} L={layer} s={strength}")
            return json.load(open(out_cell_path))

        t_start = time.time()
        # 1. Generation with steering active
        steer_vec = self.get_steer_tensor(layer, strength)
        self.set_steer(layer, steer_vec)
        gen_texts = self.generate(prompts)

        # 2. MMLU with steering active
        mmlu_preds = self.run_mmlu(mmlu_items)
        mmlu_correct = sum(pr == it["answer"] for pr, it in zip(mmlu_preds, mmlu_items))
        mmlu_acc = mmlu_correct / len(mmlu_items)

        # 3. Turn off steering for scoring
        self.set_steer(None, None)

        # 4. Mechanical scoring (PPL & trigrams)
        ppl_metrics = self.text_ppl(gen_texts)

        # 5. Judges (Topic, Fluent, Relevant)
        q_topic = "Does this text mention or talk about the Eiffel Tower or Paris?"
        q_fluent = "Ignoring what topic it is about, is this text fluent, grammatical, sensible English?"
        q_relevant = "Does the response actually address the user's question?"

        judge_topics = self.judge(gen_texts, q_topic)
        judge_fluents = self.judge(gen_texts, q_fluent)
        judge_relevants = self.judge(gen_texts, q_relevant)

        items_scored = []
        topic_count = 0
        fluent_count = 0
        relevant_count = 0
        useful_count = 0

        for p, g, pp, jt, jf, jr in zip(prompts, gen_texts, ppl_metrics, judge_topics, judge_fluents, judge_relevants):
            regex_topic = bool(RE_EIFFEL.search(g))
            judge_topic_bin = jt > 0.5
            topic_adherent = regex_topic or judge_topic_bin

            # Fluency: judge fluent AND rep_trigram_ratio <= 0.20 AND finite reasonable ppl
            is_fluent = (jf > 0.5) and (pp["rep_trigram_ratio"] <= 0.20) and (pp["ppl"] < 50.0)
            is_relevant = jr > 0.5
            is_useful = is_fluent and is_relevant

            if topic_adherent:
                topic_count += 1
            if is_fluent:
                fluent_count += 1
            if is_relevant:
                relevant_count += 1
            if is_useful:
                useful_count += 1

            items_scored.append({
                "prompt": p,
                "output": g,
                "ppl": pp["ppl"],
                "rep_trigram_ratio": pp["rep_trigram_ratio"],
                "regex_topic": regex_topic,
                "judge_topic": jt,
                "judge_fluent": jf,
                "judge_relevant": jr,
                "is_topic": topic_adherent,
                "is_fluent": is_fluent,
                "is_relevant": is_relevant,
                "is_useful": is_useful
            })

        n_p = len(prompts)
        summary = {
            "layer": layer,
            "strength": strength,
            "n_prompts": n_p,
            "topic_rate": topic_count / n_p,
            "fluent_rate": fluent_count / n_p,
            "relevant_rate": relevant_count / n_p,
            "useful_rate": useful_count / n_p,
            "median_ppl": float(sorted(m["ppl"] for m in ppl_metrics)[n_p // 2]),
            "mmlu_n": len(mmlu_items),
            "mmlu_acc": mmlu_acc,
            "elapsed_sec": time.time() - t_start,
            "items": items_scored
        }

        with open(out_cell_path, "w") as f:
            json.dump(summary, f, indent=1)

        self.log(
            f"Done {prefix} L={layer} s={strength} ({summary['elapsed_sec']:.1f}s) | "
            f"Topic={summary['topic_rate']*100:.1f}% MMLU={summary['mmlu_acc']*100:.1f}% "
            f"Relevant={summary['relevant_rate']*100:.1f}% Useful={summary['useful_rate']*100:.1f}% "
            f"MedPPL={summary['median_ppl']:.2f}"
        )
        return summary

    def run_pilot(self):
        self.log("=== STARTING PILOT RUN ===")
        prompts = self.data["pilot_prompts"]
        mmlu_items = self.data["pilot_mmlu"]

        pilot_cells = [(0, 0)]  # baseline
        for L in PILOT_LAYERS:
            for s in PILOT_STRENGTHS:
                pilot_cells.append((L, s))

        results = []
        for L, s in pilot_cells:
            res = self.evaluate_cell(L, s, prompts, mmlu_items, prefix="pilot")
            # store summary without heavy items list
            summary = {k: v for k, v in res.items() if k != "items"}
            results.append(summary)

        with open(os.path.join(self.out_dir, "pilot_summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(self.out_dir, "pilot_DONE"), "w") as f:
            f.write(f"{time.time() - self.t0:.1f}\n")
        self.log("=== PILOT RUN COMPLETE ===")

    def run_confirm(self):
        self.log("=== STARTING CONFIRMATORY RUN ===")
        prompts = self.data["confirm_prompts"]
        mmlu_items = self.data["confirm_mmlu"]

        cells = [(0, 0)]  # shared baseline
        for L in LAYERS_ALL:
            for s in STRENGTHS_ALL:
                cells.append((L, s))

        results = []
        for L, s in cells:
            res = self.evaluate_cell(L, s, prompts, mmlu_items, prefix="confirm")
            summary = {k: v for k, v in res.items() if k != "items"}
            results.append(summary)

        with open(os.path.join(self.out_dir, "confirm_summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(self.out_dir, "confirm_DONE"), "w") as f:
            f.write(f"{time.time() - self.t0:.1f}\n")
        self.log("=== CONFIRMATORY RUN COMPLETE ===")


if __name__ == "__main__":
    args = parse_args()
    runner = Runner(args.workdir)
    if args.mode == "vectors_only":
        runner.log("Vectors extracted successfully.")
    elif args.mode == "pilot":
        runner.run_pilot()
    elif args.mode == "confirm":
        runner.run_confirm()
