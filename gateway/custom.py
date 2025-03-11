import os
from typing import Iterator

from datarobot_drum import RuntimeParameters
from openai import AzureOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionCreateParams,
)
import pandas as pd


def load_model(*args, **kwargs) -> AzureOpenAI:
    openai_api_key = RuntimeParameters.get("OPENAI_API_KEY")["apiToken"]
    openai_endpoint = RuntimeParameters.get("OPENAI_ENDPOINT")
    azure_deployment_id = RuntimeParameters.get("AZURE_DEPLOYMENT_ID")
    openai_api_version = RuntimeParameters.get("OPENAI_API_VERSION")
    gateway_token = RuntimeParameters.get("GATEWAY_ACCESS_TOKEN")["apiToken"]
    gateway_endpoint = RuntimeParameters.get("GATEWAY_ENDPOINT")

    return AzureOpenAI(
        api_key=openai_api_key,
        azure_endpoint=gateway_endpoint, 
        azure_deployment = azure_deployment_id,
        api_version = openai_api_version, 
        default_headers = {"Authorization": f"Bearer {gateway_token}"}
        )


def score(data: pd.DataFrame, model, **kwargs):
    prompts = data["promptText"].tolist()
    responses = []

    for prompt in prompts:
        response = model.chat.completions.create(
            model=RuntimeParameters.get("AZURE_DEPLOYMENT_ID"),
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ],
            temperature=0,
        )
        responses.append(response.choices[0].message.content)

    return pd.DataFrame({"resultText": responses})


def chat(
    completion_create_params: CompletionCreateParams, model: AzureOpenAI
) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    # if completion_create_params["model"] == "datarobot-deployed-llm":
    completion_create_params["model"] = RuntimeParameters.get("AZURE_DEPLOYMENT_ID")
    return model.chat.completions.create(**completion_create_params)
