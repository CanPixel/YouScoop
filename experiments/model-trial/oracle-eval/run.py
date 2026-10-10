"""Run the fixed Oracle question set against an isolated archive.

Usage: run.py ORACLE_CHECK LIBRARY MODELS_DIR OUTPUT_JSON [QUESTIONS_JSON]
The first question applies corrections.json (the human Ball -> Baal spelling).
A question may set "lens" ("archive" or "esoteric"), "spoken" (true) and
"history" ([[question, answer], ...]) to exercise the app's other consultations."""
import json, os, pathlib, subprocess, sys, tempfile

here = pathlib.Path(__file__).parent
binary, library, models, output = sys.argv[1:5]
questions = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else here / "questions.json"
results = []
for i, q in enumerate(json.loads(questions.read_text())):
    env = dict(os.environ)
    env.pop("ORACLE_LENS", None); env.pop("ORACLE_SPOKEN", None); env.pop("ORACLE_HISTORY", None)
    if q.get("lens"):
        env["ORACLE_LENS"] = q["lens"]
    if q.get("spoken"):
        env["ORACLE_SPOKEN"] = "1"
    if q.get("history"):
        turns = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(q["history"], turns); turns.close()
        env["ORACLE_HISTORY"] = turns.name
    args = [binary, library, f"{models}/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            f"{models}/bge-small-en-v1.5-q8_0.gguf", q["question"]]
    if i == 0:
        args.append(str(here / "corrections.json"))
    run = subprocess.run(args, capture_output=True, text=True, check=True, env=env)
    d = json.loads(run.stdout)
    results.append({"id": q["id"], "question": q["question"], "expect": q["expect"],
                    "lens": q.get("lens", "archive"), "spoken": bool(q.get("spoken")), "threads": d.get("threads", []),
                    "answer": d["answer"], "words": len(d["answer"].split()),
                    "elapsedSeconds": round(d["elapsedSeconds"], 2),
                    "firstSentenceSeconds": round(d["spokenSentences"][0][0], 2) if d["spokenSentences"] else None,
                    "spokenSentences": [s for _, s in d["spokenSentences"]],
                    "sources": [{"start": h["evidence"]["start"], "text": h["evidence"]["text"][:200]}
                                for h in d["sources"]]})
    print(f"== {q['id']} {results[-1]['elapsedSeconds']}s, first sentence {results[-1]['firstSentenceSeconds']}s, {results[-1]['words']} words\n{d['answer']}\n", flush=True)
pathlib.Path(output).write_text(json.dumps(results, indent=1))
