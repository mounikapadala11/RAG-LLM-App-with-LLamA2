# Retrieval-Augmented Generation (RAG) System Using LLaMA 2

This project is a retrieval-augmented generation (RAG) system using open-source models like LLaMA 2. The system is designed to load, index, and query PDF documents effectively, leveraging advanced language models and embedding techniques.

## Introduction and Objective

The goal of this project is to create a robust RAG system using open-source models, specifically LLaMA 2. The system will:

- Load and index PDF documents.
- Allow efficient querying of the documents using LLaMA 2 from Hugging Face.

## Setup and Installation

### Google Colab

For implementation, I used Google Colab.

### Libraries

The following libraries are required for the project:

```bash
pip install pypdf transformers einops accelerate langchain bitsandbytes sentence-transformers llama-index llama-index-llms-huggingface
```

### Loading and Indexing PDFs

PDF documents are loaded from a 'data' folder using SimpleDirectoryReader from LLaMA Index. This step involves extracting text from PDFs and indexing them for efficient querying.

```bash
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data").load_data()
```

### Quantization

To optimize the model for running on limited resources like those available in Google Colab, I used quantization. This process reduces the model size from 16-bit to 4-bit, significantly improving efficiency.

```bash
import bitsandbytes as bnb
```

## Model Setup

### Hugging Face Login

Ensure you are logged into Hugging Face to access the LLaMA 2 model:

```bash
!huggingface-cli login
```

### Model Configuration

The LLaMA 2 model (7 billion parameters) is configured with the following settings:

- Context window size of 4096.
- Temperature setting to control creativity.
- Quantization to 8-bit using `load_in_8bit=True.`

```bash
import torch
from llama_index.llms.huggingface import HuggingFaceLLM

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
)
```

## Embedding Model

The `sentence-transformers` model `all-mpnet-base-v2` is used for embeddings. This model maps sentences and paragraphs to a 768-dimensional dense vector space.

```bash
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
```

## Service Context

A `ServiceContext` is created to bundle the LLM model, embedding model, prompts, and documents for indexing and querying. The `ServiceContext` includes:

- Chunk size.
- LLM model.
- Embedding model.

```bash
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=1024)
```

## Indexing and Querying

The documents are converted into a `VectorStoreIndex` using the `ServiceContext`. This index is then converted into a `QueryEngine` to enable querying.

```bash
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents, embed_model=Settings.embed_model
)

query_engine = index.as_query_engine(llm=Settings.llm)

```

## Prompt 

A system prompt and a query wrapper prompt are created to provide instructions and context for the LLaMA 2 model.

```bash
from llama_index.core.prompts.prompts import SimpleInputPrompt

system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
query_wrapper_prompt = SimpleInputPrompt("{query_str}")
```

## Execution and Results

The model retrieves responses based on the indexed documents, showcasing the system's capabilities.

Developed a Retrieval-Augmented Generation (RAG) system using LLaMA 2, integrating PDF document indexing and querying with quantization techniques to optimize performance on limited resources. Implemented embeddings with sentence-transformers and combined LLaMA Index and LangChain for efficient document retrieval and question-answering capabilities.

