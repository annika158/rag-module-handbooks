import os
import pickle
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

###### PREPARATION FOR THE MODULE HANDBOOK RAG-SYSTEM (LOADING, PROCESSING, SPLITTING, EMBEDDING, ...) ######

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoids parallelism error in embedding

# Set API key and models
api_key = os.getenv("OPENAI_API_KEY")
base_url = "https://gpt.uni-muenster.de/v1"
model = "Llama-3.1-70B"  # Options: "Llama-3.1-70B", "mixtral-8x7B"
huggingface_model = "sentence-transformers/all-mpnet-base-v2"  # example alternative: "intfloat/multilingual-e5-small"

# TODO: Change the names of the directory and collection as needed/wanted
persist_directory = "vectorstores"
collection_name = "module_handbooks_economics_unmodified_chunks"

os.environ["OPENAI_API_BASE"] = base_url
os.environ["OPENAI_API_KEY"] = api_key

def extract_metadata_from_first_page(first_page_text):
    """
    Extract the metadata (study program, degree, effective from, examination regulation) from the first page using an LLM.
    """
    response_schemas = [
        ResponseSchema(name="study_program", description="Name of the study program"),
        ResponseSchema(name="degree", description="Degree, either Bachelor or Master"),
        ResponseSchema(name="effective_from", description="Effective from date or semester"),
        ResponseSchema(name="examination_regulation", description="Examination regulation (PO)")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = """
    Extract the following information from the text below:

    - study_program
    - degree (Bachelor or Master)
    - effective_from
    - examination_regulation (PO)

    If any information is missing, output "unknown" for that field.

    Provide the output in the following JSON format:
    {format_instructions}

    Text:
    {first_page_text}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model_name=model, temperature=0)
    chain = prompt | llm | output_parser

    inputs = {
        "first_page_text": first_page_text,
        "format_instructions": format_instructions
    }

    try:
        extracted_info = chain.invoke(inputs)
    except Exception as e:
        print(f"Error processing first page: {e}")
        extracted_info = {
            "study_program": "unknown",
            "degree": "unknown",
            "effective_from": "unknown",
            "examination_regulation": "unknown"
        }

    return extracted_info

def load_and_process_pdfs(pdf_directory):
    """
    Load each module handbook (one PDF) as one Document from a directory and extract the metadata.
    """
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_directory, filename)
            # Initialize the PyPDFLoader for the PDF
            loader = PyPDFLoader(file_path)
            # Load all pages as separate documents
            pages = loader.load()
            # Combine the content of all pages into one string
            full_text = '\n'.join([page.page_content for page in pages])

            # Extract the metadata from the first page
            first_page_text = pages[0].page_content
            extracted_info = extract_metadata_from_first_page(first_page_text)

            # Create a new Document with combined text and metadata
            combined_doc = Document(
                page_content=full_text,
                metadata={
                    "source": file_path,
                    "study_program": extracted_info.get("study_program", "unknown"),
                    "degree": extracted_info.get("degree", "unknown"),
                    "effective_from": extracted_info.get("effective_from", "unknown"),
                    "examination_regulation": extracted_info.get("examination_regulation", "unknown")
                }
            )
            # Append the combined document to the list
            documents.append(combined_doc)
    return documents

def split_documents_into_modules_and_chunks(documents):
    """
    Split documents into modules using regex and then into smaller chunks based on tokens.
    """
    def split_document_into_modules(document_text):
        # Split based on module titles
        # For the economics module handbooks, the module title is either "Modultitel deutsch:" or "Module Title english:"
        modules = re.split(r'\n\s*(?=Modultitel deutsch\s*:|Module Title english\s*:)', document_text, flags=re.IGNORECASE)
        return modules

    modules = []
    for doc in documents:
        modules_text = split_document_into_modules(doc.page_content)
        # Extract module title and add to metadata
        for module_text in modules_text:
            module_title_match = re.search(r'(Modultitel deutsch|Module title english)\s*:\s*(.+)', module_text, flags=re.IGNORECASE)
            module_title = module_title_match.group(2) if module_title_match else 'General Information or Unknown Module'
            metadata = doc.metadata.copy()
            metadata["module_title"] = module_title

            module = Document(
                page_content=module_text,
                metadata=metadata
            )
            modules.append(module)

    # Split the modules into chunks based on tokens and optionally insert at the top of each chunk the metadata of the module
        text_splitter = SentenceTransformersTokenTextSplitter(
        model_name=huggingface_model,
        chunk_overlap=50)
    
    chunks = []
    for module in modules:
        metadata_str = (
        f"Study Program: {module.metadata['study_program']}\n"
        f"Degree: {module.metadata['degree']}\n"
        f"Examination Regulation: {module.metadata['examination_regulation']}\n"
        f"Effective From: {module.metadata['effective_from']}\n"
        f"Module Title: {module.metadata['module_title']}\n"
        "\n")

        # TODO: To create chunks with the metadata prepended, comment the following line
        # (This is not ideal, but this is how I created chunks_modified.pkl and chunks_unmodified.pkl, and to ensure the exact same chunks as used for the thesis results, this will be left as it is)
        metadata_str = "" 

        # Adjust tokens_per_chunk to fit the metadata
        metadata_token_count = text_splitter.count_tokens(text=metadata_str)
        adjusted_tokens_per_chunk = text_splitter.maximum_tokens_per_chunk - metadata_token_count
        if adjusted_tokens_per_chunk <= 0:
            raise ValueError(f"Metadata is too long! Token count: {metadata_token_count}")
        text_splitter.tokens_per_chunk = adjusted_tokens_per_chunk

        # Split module content into chunks
        chunked_module = text_splitter.split_documents([module])
        # Create new Documents for each chunk with metadata prepended
        for chunk in chunked_module:
            chunk.page_content = metadata_str + chunk.page_content
            chunks.append(chunk)
    return chunks

def save_documents(documents, filename):
    """
    Save documents to a pickle file.
    """
    with open(filename, "wb") as f:
        pickle.dump(documents, f)

def load_documents(filename):
    """
    Load documents from a pickle file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def main():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model)

    # The prepossessing can be done in multiple steps (with saving and loading the documents to/from a file) with uncommenting/commenting the blocks step by step
    # by uncommenting every block the preprocessing will be done from scratch in one step (still with saving and loading the documents to/from a file; might take some time)

    # Uncomment the following block to load and process the module handbooks and save them as documents
    # If no changes were made to the module handbooks, this block can be skipped as the results are already saved in module_handbooks_economics.pkl
    """ 
    print("Processing PDFs...")
    pdf_directory = 'module_handbooks_economics'
    documents = load_and_process_pdfs(pdf_directory)
    save_documents(documents, "module_handbooks_economics.pkl")
    """

    # Uncomment the following block if the module handbooks have already been processed and saved
    """ 
    documents = load_documents("module_handbooks_economics.pkl")
    """
    
    # Uncomment the following block to split the documents into chunks 
    # If no changes were made to the module handbooks, this block can be skipped as the results are already saved in chunks_unmodified.pkl and chunks_modified.pkl
    """ 
    print("Splitting documents into modules and chunks...")
    chunks = split_documents_into_modules_and_chunks(documents)
    save_documents(chunks, "chunks_unmodified.pkl")
    """

    # Uncomment the following block to load the saved chunks and save them to a local vectorstore
    
    chunks = load_documents("chunks_unmodified.pkl")
    print("Chunks loaded from file, now saving to vectorstore...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    ) 
    print("Saved to vectorstore.")


if __name__ == "__main__":
    main()