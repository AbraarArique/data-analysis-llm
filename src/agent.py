from datetime import datetime
from langchain_core.runnables import chain
from langchain_core.messages import ToolMessageChunk
from langchain_core.runnables.history import RunnableWithMessageHistory
from chat import get_session_history
from tools import limit, template, model, toolsHash

# Core chain and chat history
chain = limit | template | model
with_history = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="messages"
)


def agent(input, config):
    while True:
        response = None

        # Gather streamed response
        for chunk in with_history.stream(input, config=config):
            # Yield chunks for streaming support
            yield chunk

            if response is None:
                response = chunk
            else:
                response += chunk

        # Call each requested tool
        if response.tool_calls:
            messages = []
            for call in response.tool_calls:
                tool = toolsHash.get(call["name"])
                if tool:
                    output = tool.invoke(call["args"])
                    tool_message = ToolMessageChunk(output, tool_call_id=call["id"])
                    yield tool_message
                    messages.append(tool_message)
        else:
            break
