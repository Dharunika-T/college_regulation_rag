import sys

from src.agents import answer_question
from src.ingest import index_pdfs


def chat():
    print("Chat started. Type 'exit' to quit.")
    while True:
        try:
            question = input("User: ").strip()
        except EOFError:
            break

        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print(f"Agent: {answer_question(question)}\n")


def main(args=None):
    args = sys.argv[1:] if args is None else args

    if not args or args[0] == "chat":
        chat()
    elif args[0] == "ingest":
        counts = index_pdfs()
        print(f"Indexed {sum(counts.values())} chunks:")
        for filename, count in counts.items():
            print(f"{filename}: {count}")
    elif args[0] == "query":
        question = " ".join(args[1:])
        print(f"Agent: {answer_question(question)}")
    else:
        print('Use: python -m src.main ingest OR python -m src.main query "your question"')


if __name__ == "__main__":
    main()
