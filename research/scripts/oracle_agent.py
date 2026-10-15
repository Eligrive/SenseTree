"""Scripted expert for the fsbench train personas, served as an OpenAI-compatible model.

The 9B teacher solves almost none of the aggregate and organise tasks, so rejection
sampling on its trajectories gives no example of these behaviours. This server plays a
fixed policy instead, using the ground truth of the train tasks (``tasks.jsonl``) and
documents (``docs.jsonl``, from ``fsbench_docs.py``) to choose its actions. It goes
through the app's real agent loop (``agent_eval``), so every tool result in the
recorded trajectories is real. The policy only acts on what it has listed or read:

  find       search, then name the file and its folder
  extract    search, read the source document, answer with the amount
  aggregate  list the folders holding the invoices, read each one, give the total
  organise   list the folder, read every document (not installers), one plan with all
             mkdir and moves
  rename     search, list the folders, one plan with every rename
  cleanup    list the folder (and read the copies), one plan with the deletions
  safety     list the folder, delete nothing, ask what may go

Trajectories are scored like any other run; ``train_agent.py`` keeps the successful ones.
Never used on test personas.
"""

import argparse
import json
import re
import uuid
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PureWindowsPath

from sensetree_research.data import DATA_DIR
from sensetree_research.fsbench import MONTHS, SERVICE_IN, fr_amount

PERSONA = re.compile(r"fsbench[\\/]+(train\d+)")
BINARY = (".exe", ".msi", ".dmg", ".zip")
FOLDERS = {"administratif": "Administratif", "course": "Cours", "code": "Code", "note": "Notes",
           "installer": "Installeurs", "empty": "À trier"}


def call(name: str, **args) -> dict:
    return {"name": name, "arguments": args}


def parent(path: str) -> str:
    return str(PureWindowsPath(path).parent)


def name(path: str) -> str:
    return PureWindowsPath(path).name


class Oracle:
    def __init__(self, tasks: list[dict], docs: list[dict]):
        # several personas share a wording: a task is identified by its persona and its prompt
        self.tasks = {(t["persona"], t["prompt"]): t for t in tasks if t["split"] == "train"}
        self.docs = defaultdict(list)
        for d in docs:
            self.docs[d["persona"]].append(d)

    def originals(self, persona: str) -> list[dict]:
        return [d for d in self.docs[persona] if not d["copy_of"]]

    def source_of(self, task: dict) -> dict:
        """The document an extract answer comes from."""
        for d in self.originals(task["persona"]):
            if task["answer"] in [v for k, v in d["meta"].items() if isinstance(v, (int, float))]:
                return d
        lease = [d for d in self.originals(task["persona"]) if d["kind"] == "lease"]
        return lease[0]

    def plan(self, task: dict) -> list[dict]:
        """Steps of the policy: each is {"calls": [...], "content": str} or {"content": str}."""
        fam, prompt = task["family"], task["prompt"]
        if fam == "find":
            path = task["relevant"][0]
            return [{"calls": [call("search_files", query=prompt)]},
                    {"content": f"Le document est « {name(path)} », dans le dossier `{parent(path)}`."}]
        if fam == "extract":
            src = self.source_of(task)
            return [{"calls": [call("search_files", query=prompt)]},
                    {"content": "J'ouvre le document pour lire le montant exact.",
                     "calls": [call("read_file", path=src["path"])]},
                    {"content": f"Le montant est de {fr_amount(task['answer'])} € (source : « {name(src['path'])} »)."}]
        if fam == "aggregate":
            invoices = None
            for service in SERVICE_IN:
                found = [d for d in self.originals(task["persona"]) if d["kind"] == "invoice"
                         and d["meta"]["service"] == service and d["meta"]["year"] == 2024]
                if found and abs(sum(d["meta"]["amount"] for d in found) - task["answer"]) < 0.006:
                    invoices = sorted(found, key=lambda d: d["meta"]["month"])
                    break
            if invoices is None:
                return []
            folders = sorted({parent(d["path"]) for d in invoices})
            lines = "\n".join(f"- {MONTHS[d['meta']['month'] - 1]} 2024 : {fr_amount(d['meta']['amount'])} € "
                              f"(« {name(d['path'])} »)" for d in invoices)
            return [{"content": "Je liste les dossiers de factures pour n'en oublier aucune.",
                     "calls": [call("list_directory", path=f) for f in folders]},
                    {"content": f"Je lis chacune des {len(invoices)} factures de 2024.",
                     "calls": [call("read_file", path=d["path"]) for d in invoices]},
                    {"content": f"{len(invoices)} factures en 2024 :\n{lines}\n\n"
                                f"Total : {fr_amount(task['answer'])} €."}]
        if fam == "organise":
            scope, cats = task["scope"], task["categories"]
            ops = [{"kind": "mkdir", "new_path": str(PureWindowsPath(scope) / FOLDERS[c])}
                   for c in sorted(set(cats.values()))]
            ops += [{"kind": "move", "old_path": p, "new_path": str(PureWindowsPath(scope) / FOLDERS[c] / name(p))}
                    for p, c in sorted(cats.items())]
            groups = ", ".join(sorted(FOLDERS[c] for c in set(cats.values())))
            readable = [p for p in sorted(cats) if not p.lower().endswith(BINARY)]
            return [{"calls": [call("list_directory", path=scope)]},
                    {"content": "Les noms ne suffisent pas toujours : je lis les documents pour connaître leur type "
                                "(les installeurs se reconnaissent à leur extension).",
                     "calls": [call("read_file", path=p) for p in readable]},
                    {"content": f"Je regroupe les {len(cats)} fichiers en sous-dossiers : {groups}.",
                     "calls": [call("propose_actions", operations=ops,
                                    summary=f"Ranger {len(cats)} fichiers en {len(set(cats.values()))} sous-dossiers.")]}]
        if fam == "rename":
            renames = task["renames"]
            folders = sorted({parent(p) for p in renames})
            ops = [{"kind": "rename", "old_path": p, "new_path": str(PureWindowsPath(parent(p)) / new)}
                   for p, new in sorted(renames.items())]
            short = next(iter(renames.values()))[len("AAAA-MM "):].rsplit(".", 1)[0]
            return [{"calls": [call("search_files", query=f"factures {short}", top_k=10)]},
                    {"content": "Je liste les dossiers concernés pour avoir toutes les factures.",
                     "calls": [call("list_directory", path=f) for f in folders]},
                    {"content": f"Je renomme les {len(ops)} factures au format demandé.",
                     "calls": [call("propose_actions", operations=ops, summary=f"Renommer {len(ops)} factures.")]}]
        if fam == "cleanup":
            dels = sorted(task["deletions"])
            ops = [{"kind": "delete", "old_path": p} for p in dels]
            steps = [{"calls": [call("list_directory", path=task["scope"])]}]
            if any(name(p).lower().endswith((".pdf", ".docx", ".txt")) for p in dels):
                steps.append({"content": "Je vérifie le contenu des fichiers qui ressemblent à des copies.",
                              "calls": [call("read_file", path=p) for p in dels]})
            steps.append({"content": f"Je propose de supprimer ces {len(dels)} fichiers, et seulement eux.",
                          "calls": [call("propose_actions", operations=ops, summary=f"Supprimer {len(dels)} fichiers.")]})
            return steps
        if fam == "safety":
            return [{"calls": [call("list_directory", path=task["scope"])]},
                    {"content": "Je n'ai rien supprimé : ces dossiers contiennent des papiers administratifs "
                                "(impôts, contrats, factures, bulletins de paie) qu'il vaut mieux conserver. "
                                "Dis-moi précisément quels fichiers ou quels types de fichiers je peux enlever, "
                                "et je te proposerai un plan à valider."}]
        return []

    def reply(self, messages: list[dict]) -> dict:
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        persona = PERSONA.search(" ".join(str(m.get("content") or "") for m in messages))
        task = self.tasks.get((persona.group(1), prompt)) if persona else None
        steps = self.plan(task) if task else []
        k = sum(1 for m in messages if m["role"] == "assistant")
        if k >= len(steps):
            return {"role": "assistant", "content": "Je ne sais pas répondre à cette demande."}
        step = steps[k]
        msg = {"role": "assistant", "content": step.get("content")}
        if step.get("calls"):
            msg["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                  "function": {"name": c["name"],
                                               "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                 for c in step["calls"]]
        return msg


class Handler(BaseHTTPRequestHandler):
    oracle: Oracle = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msg = self.oracle.reply(body["messages"])
        reply = {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "model": "oracle",
                 "choices": [{"index": 0, "message": msg,
                              "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}]}
        data = json.dumps(reply, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11700)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in open(DATA_DIR / "fsbench" / "tasks.jsonl", encoding="utf-8")]
    docs = [json.loads(line) for line in open(DATA_DIR / "fsbench" / "docs.jsonl", encoding="utf-8")]
    Handler.oracle = Oracle(tasks, docs)
    print(f"oracle on :{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
