#!/usr/bin/env python3
"""
YouTube RAG v2 - Multilingual Semantic Search

Multi-video semantic search with RAG-powered answer generation.
Supports Japanese and English transcripts (extensible to zh/ko).

Commands:
  add <url> [language]  Add a YouTube video (language: ja, en; default: ja)
  list                  List all indexed videos
  remove <id>           Remove a video from the library
  search <query>        Semantic search across all videos
  ask <question>        Ask a question (RAG: search + LLM answer)
  save                  Save library to disk
  quit                  Exit the program
"""

import sys

from multilingual.rag_engine import RAGEngine
from multilingual.text_processing import LANGUAGE_CONFIG


def display_search_results(results):
    """Pretty-print search results with video source attribution."""
    if not results:
        print("\nNo results found.")
        return

    for r in results:
        lang_tag = f"[{r.get('language', '??')}]"
        print(f"\n{'='*80}")
        print(f"Rank #{r['rank']} | Score: {r['score']:.4f} | {r['video_title']} {lang_tag}")
        print(f"Time: {r['start']:.1f}s - {r['end']:.1f}s")
        print(f"URL: {r['url']}")
        print(f"{'-'*80}")
        text = r["text"]
        print(f"{text[:500]}{'...' if len(text) > 500 else ''}")
        print(f"{'='*80}")


def display_rag_answer(result):
    """Pretty-print a RAG answer with citations."""
    print(f"\n{'='*80}")
    print("Answer:")
    print(f"{'='*80}")
    print(result["answer"])
    print(f"\n{'-'*80}")
    print(f"Sources ({len(result['sources'])} chunks, model: {result['model']}):")
    for s in result["sources"]:
        start_min = int(s["start"] // 60)
        start_sec = int(s["start"] % 60)
        lang_tag = f"[{s.get('language', '??')}]"
        print(f"  - {s['video_title']} {lang_tag} @ {start_min}:{start_sec:02d} ({s['url']})")
    print(f"{'='*80}")


def main():
    """Main interactive CLI loop."""
    print("=" * 80)
    print("YouTube RAG v2 - Multilingual Semantic Search")
    print(f"Supported languages: {', '.join(LANGUAGE_CONFIG.keys())}")
    print("=" * 80)

    try:
        engine = RAGEngine()
    except Exception as e:
        print(f"Error initializing: {e}")
        print("Make sure you have installed required packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    lib = engine.library
    n_videos = len(lib.videos)
    if n_videos > 0:
        print(f"\nLoaded library with {n_videos} video(s), {lib.total_chunks} chunks.")
    else:
        print("\nNo saved library found. Add videos with 'add <url> [language]'.")

    print("\nCommands: add <url> [lang] | list | remove <id> | search <query> | ask <question> | save | quit")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        elif command == "add":
            if not arg:
                print("Usage: add <youtube_url> [language]")
                print(f"  Languages: {', '.join(LANGUAGE_CONFIG.keys())} (default: ja)")
                continue
            # Parse optional language argument
            add_parts = arg.split()
            url = add_parts[0]
            language = add_parts[1] if len(add_parts) > 1 else "ja"
            if language not in LANGUAGE_CONFIG:
                print(f"Unknown language '{language}'. Available: {', '.join(LANGUAGE_CONFIG.keys())}")
                continue
            try:
                lib.add_video(url, language=language)
            except Exception as e:
                print(f"Error adding video: {e}")

        elif command == "list":
            videos = lib.list_videos()
            if not videos:
                print("Library is empty. Add videos with 'add <url> [language]'.")
            else:
                print(f"\n{len(videos)} video(s) in library:")
                for v in videos:
                    print(f"  [{v['video_id']}] {v['title']} [{v['language']}] ({v['num_chunks']} chunks)")

        elif command == "remove":
            if not arg:
                print("Usage: remove <video_id>")
                continue
            try:
                lib.remove_video(arg)
            except KeyError as e:
                print(f"Error: {e}")

        elif command == "search":
            if not arg:
                print("Usage: search <query>")
                continue
            try:
                results = engine.search(arg, k=5)
                display_search_results(results)
            except Exception as e:
                print(f"Search error: {e}")

        elif command == "ask":
            if not arg:
                print("Usage: ask <question>")
                continue
            try:
                result = engine.ask(arg, k=5)
                display_rag_answer(result)
            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error generating answer: {e}")

        elif command == "save":
            try:
                lib.save()
            except Exception as e:
                print(f"Error saving: {e}")

        else:
            print(f"Unknown command: {command}")
            print("Commands: add <url> [lang] | list | remove <id> | search <query> | ask <question> | save | quit")


if __name__ == "__main__":
    main()
