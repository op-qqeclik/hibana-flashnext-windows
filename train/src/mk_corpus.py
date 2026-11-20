"""Turn the local {bucket, messages} corpus into the recipe's {input_ids, prompt_len, domain, thinking} rows."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/qqelick/hibana")
from hibana.api.chat_template import ChatTemplate
from hibana.tokenizer import SnapshotTokenizer
from hibana.target import _flashnext as target

snap = target().snapshot_dir()
tmpl, tok = ChatTemplate(snap), SnapshotTokenizer(snap)
out = open("/home/qqelick/mtp_train/corpus13.jsonl", "w")
n = 0
for line in Path("/home/qqelick/mtp_train/corpus/balanced3.jsonl").read_text().splitlines():
    r = json.loads(line)
    msgs = r["messages"]
    ids = tok.encode(tmpl.render(msgs, enable_thinking=False))
    head = tok.encode(tmpl.render(msgs[:-1], enable_thinking=False))
    if len(ids) < len(head) + 16 or not ids[:len(head)] == head:
        continue
    out.write(json.dumps({"domain": r["bucket"], "thinking": False, "input_ids": ids,
                          "prompt_len": len(head)}, ensure_ascii=False) + "\n")
    n += 1
print("rows", n)
