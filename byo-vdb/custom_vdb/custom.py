import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import torch
import json
from datarobot_drum import RuntimeParameters
import os
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


def load_model(input_dir, **kwargs):

    if not torch.cuda.is_available():
        EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    else:
        EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=os.path.join(input_dir, "storage/deploy/sentencetransformers"),
    )
    vectordb = Chroma(
        persist_directory=os.path.join(input_dir, "chroma/datarobot_task_catalog"),
        embedding_function=embedding_function,
    )
    retriever = vectordb.as_retriever()
    retriever.search_kwargs["k"] = 10
    return retriever


def score_unstructured(retriever, data, query, **kwargs):
    headers = kwargs.get("headers")
    try:
        data_dict = json.loads(data)
        query = data_dict["question"]
        relevant_docs = retriever.invoke(query)
        rv = {
            "question": query,
            "relevant": [r.page_content for r in relevant_docs],
            "references": [r.metadata for r in relevant_docs],
        }
    except Exception as e:
        print(e)
        rv = {"error": str(e)}

    return json.dumps(rv), {"mimetype": "application/json", "charset": "utf8"}
