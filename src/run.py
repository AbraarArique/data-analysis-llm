from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from agent import agent
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("message", help="Add a message to the AI chat thread")
    parser.add_argument("--lang", help="LLM response language", default="English")
    parser.add_argument("--thread", help="An ID for chat thread/history", default="cat")
    args = parser.parse_args()

    message = args.message
    config = {"configurable": {"session_id": args.thread}}

    res = None
    stream = RunnableLambda(agent).stream(
        {"messages": [HumanMessage(message)], "language": "English"}, config
    )

    for chunk in stream:
        if res is None:
            res = chunk
        else:
            res += chunk

        if chunk.content:
            print(chunk.content, end="")

    # print(res.content)
