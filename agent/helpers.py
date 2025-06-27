from typing import List
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.beta.thread import Thread
from openai.types.beta.assistant import Assistant
import time
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
import uuid

def convert_run_result_to_chat_completion(run_result) -> ChatCompletion:
    tool_calls = []
    for tool in run_result.new_items:
        if isinstance(tool.raw_item, ResponseFunctionToolCall):
            # Properly format the function call
            tool_calls.append({
                "id": str(uuid.uuid4()),  # Generate a unique ID for each tool call
                "type": "function",        # Must be "function"
                "function": {
                    "name": tool.raw_item.name,
                    "arguments": tool.raw_item.arguments
                }
            })

    response = {
        "id": f"chatcmpl-{str(uuid.uuid4())}",  # Add required ID field
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": run_result.final_output,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "created": int(time.time()),
        "model": "gpt-4",  # or whatever model you're using
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }
    
    return ChatCompletion(**response)