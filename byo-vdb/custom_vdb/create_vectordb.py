import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import torch
import pandas as pd 
from langchain.text_splitter import TokenTextSplitter
# from langchain.document_loaders import DataFrameLoader
from langchain_community.document_loaders import DataFrameLoader
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
import numpy as np
import pandas as pd 
import re
import pathlib

asset_path = pathlib.Path(__file__).parent.resolve()

if not torch.cuda.is_available():
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
else:
    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

embedding_function = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    cache_folder= str(asset_path / "sentence-transformer"),
)

df = pd.read_csv("task_dataset_v2.csv")
# initialize embedding engine
# embeddings = OpenAIEmbeddings()
loader = DataFrameLoader(df, page_content_column="description")
data = loader.load()
# split the documents
text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)
print(f"Number of documents for products: {len(docs)}")

db = Chroma.from_documents(
    docs,
    embedding_function,
    persist_directory= str(asset_path / "chroma/datarobot_task_catalog"),
)
db.persist()