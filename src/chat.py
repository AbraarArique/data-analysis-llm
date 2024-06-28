from langchain_community.chat_message_histories import ChatMessageHistory

store = {}


def get_session_history(id):
    if id not in store:
        store[id] = ChatMessageHistory()
    return store[id]


def limit(input):
    return {**input, "messages": input["messages"][-10:]}
