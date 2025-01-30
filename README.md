# RAG system for module handbooks

## How to Proceed
1. Create a virtual environment (optionally) and install the required packages (see below).
2. Create a `.env` file containing your API key:
    ```
    OPENAI_API_KEY = "sk-..."
    ```
3. Run `preparation.py`:
    - Adjust the vectorstore directory and collection name as wanted.
4. Run `rag_basic.py` for results **without metadata filtering**:
    - Adjust the vectorstore directory and collection name according to your setup.
5. Run `rag_metadata_filter.py` for results **with metadata filtering**:
    - Adjust the vectorstore directory and collection name according to your setup.

## Required packages
langchain <br>
langchain-community <br>
langchain-openai <br>
langchain-huggingface <br>
langchain-chroma <br>
chromadb <br>
sentence-transformers <br>
openai <br>

```bash
pip install langchain langchain-community langchain-openai langchain-huggingface langchain-chroma  
```

```bash
pip install chromadb sentence-transformers openai
```

## Other Information
- The `result_comparison` folder contains files and corresponding results that were used/documented for the thesis.
    - The python files have the same functionality as `rag_basic.py` and `rag_metadata_filter.py`, but they save the results for comparison purposes and may be structured a little different. They also may be adjusted a little (like only using the module handbook as metadata filtering or other questions for the demonstration).
    - Please ignore this folder as it is just for documentation purposes and the results can be found in the thesis in a more structured and concise way.