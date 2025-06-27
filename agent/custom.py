from agents import Agent, Runner
from helpers import convert_run_result_to_chat_completion
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    CompletionCreateParams,
    ChatCompletionChunk,
    ChatCompletionMessageToolCall
)
import asyncio 
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
import os 

os.environ["MLOPS_DEPLOYMENT_ID"] = "685d8864357e85b157722450"

try:
    print("configuring tracing")
    entity_type = "deployment"
    entity_id = os.environ["MLOPS_DEPLOYMENT_ID"]
    api_key = os.environ["DATAROBOT_API_TOKEN"]
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
        f"X-DataRobot-Entity-Id={entity_type}-{entity_id},X-DataRobot-Api-Key={api_key}"
    )
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://app.datarobot.com/otel"
    traces_endpoint = 'http://app.datarobot.com/otel/v1/traces'
    os.environ['OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'] = traces_endpoint
    endpoint = "https://app.datarobot.com/otel/v1/traces"
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    # Optionally, you can also print the spans to the console.
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    print("configuring tracing complete")
except Exception as e:
    print(e)
    print("tracing is not available")


from agent import agent
def load_model(code_dir):
    return agent
def chat(
    completion_create_params: CompletionCreateParams, model: Agent
) -> ChatCompletion:
    
    user_input = completion_create_params["messages"][-1]["content"]
    run_result = asyncio.run(Runner.run(model, user_input))
    chat_completion = convert_run_result_to_chat_completion(run_result)
    return chat_completion
