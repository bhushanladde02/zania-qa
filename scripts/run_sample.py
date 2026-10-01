"""Run the whole QA flow from the command line, no server needed.

Same code path as POST /qa: parse questions -> load doc -> index -> answer.

usage:
  python scripts/run_sample.py QUESTIONS.json DOCUMENT.(pdf|json) [-o OUT.json]

example:
  python scripts/run_sample.py tests/fixtures/sample_questions.json \\
      tests/fixtures/sample_document.json -o docs/sample_output.json
"""
import argparse
import json
import sys
from pathlib import Path

# so "app" imports work when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.loaders import LoaderError, load_document, parse_questions  # noqa: E402
from app.rag import answer_questions, build_index  # noqa: E402


def run(questions_path, document_path, llm, embeddings, settings: Settings) -> dict:
    questions = parse_questions(Path(questions_path).read_bytes())
    docs = load_document(str(document_path), Path(document_path).read_bytes())
    index = build_index(docs, embeddings, settings.chunk_size, settings.chunk_overlap)
    return {"results": answer_questions(index, questions, llm, top_k=settings.top_k)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Answer questions against a document (PDF or JSON).")
    ap.add_argument("questions", help="JSON file with the questions")
    ap.add_argument("document", help="PDF or JSON document")
    ap.add_argument("-o", "--out", help="write the result here (prints to stdout if not given)")
    args = ap.parse_args(argv)

    settings = get_settings()
    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set (put it in .env)")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)
    emb = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

    try:
        result = run(args.questions, args.document, llm, emb, settings)
    except (LoaderError, FileNotFoundError) as e:
        sys.exit(f"input error: {e}")
    except openai.AuthenticationError:
        sys.exit("OpenAI rejected the API key (check OPENAI_API_KEY in .env)")
    except openai.APIError as e:
        sys.exit(f"OpenAI error: {e}")

    out = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out)
        print(f"wrote {len(result['results'])} answers to {args.out}")
    else:
        print(out, end="")


if __name__ == "__main__":
    main()
