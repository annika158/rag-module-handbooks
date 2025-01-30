import os
from dotenv import load_dotenv
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from difflib import SequenceMatcher
from langchain.schema import Document


###### RAG-SYSTEM FOR MODULE HANDBOOKS - with metadata filtering implemented ######
# Prerequisite: module handbooks are already processed, chunked and saved in the vectorstore (see main.py)

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoids parallelism error in embedding

# Set API key and models
api_key = os.getenv("OPENAI_API_KEY")
base_url = "https://gpt.uni-muenster.de/v1"
model = "Llama-3.1-70B"  # Options: "Llama-3.1-70B", "mixtral-8x7B"
huggingface_model = "sentence-transformers/all-mpnet-base-v2" # example alternative: "intfloat/multilingual-e5-small"

os.environ["OPENAI_API_BASE"] = base_url
os.environ["OPENAI_API_KEY"] = api_key


def extract_metadata(question):
    """
    Extract the metadata (study program, degree, effective from, examination regulation, module_title) from the user query using an LLM.
    Only one module title is extracted.
    """
    response_schemas = [
        ResponseSchema(name="study_program", description="Name of the study program"),
        ResponseSchema(name="degree", description="Degree, either Bachelor or Master"),
        ResponseSchema(
            name="effective_from",
            description="Effective or starting from which semester",
        ),
        ResponseSchema(
            name="examination_regulation", description="Examination regulation (PO)"
        ),
        ResponseSchema(name="module_title", description="Name or title of the Module asked for (if multiple, choose only one). Examples include 'Bachelor's thesis', 'Operations Research', or other specific academic modules."),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = """
    Extract the following information from the user query below:

    - study_program
    - degree (Bachelor or Master)
    - effective_from
    - examination_regulation (PO)
    - module_title

    If any information is missing, output "unknown" for that field.

    Provide the output in the following JSON format:
    {format_instructions}

    User Query:
    {user_query}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model_name=model, temperature=0)
    chain = prompt | llm | output_parser

    inputs = {"user_query": question, "format_instructions": format_instructions}

    try:
        extracted_metadata = chain.invoke(inputs)
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        extracted_metadata = {
            "study_program": "unknown",
            "degree": "unknown",
            "effective_from": "unknown",
            "examination_regulation": "unknown",
            "module_title": "unknown",
        }

    return extracted_metadata


# There are two options to identify the source of the module handbook: with an LLM or with a vectorstore -> with the example questions, the LLM works better than the vectorstore
def identify_source(extracted_metadata):
    """
    Identify the source of the module handbook based on the extracted metadata.
    """
    # it is not guaranteed that the right source is identified but for the examples it works
    source = "unknown"
    with open("module_handbooks_economics_metadata.txt", "r", encoding="utf-8") as f:
        sources_string = f.read()
    
    metadata_string = f"""
    {{
        'study_program': '{extracted_metadata["study_program"]}',
        'degree': '{extracted_metadata["degree"]}',
        'effective_from': '{extracted_metadata["effective_from"]}',
        'examination_regulation': '{extracted_metadata["examination_regulation"]}'
    }}
    """

    prompt = f"""
        Based on the following METADATA and LIST, identify the module handbook source that best fits according to the rules below and return ONLY the name of the source.

        Rules:
        - Do not consider the date in the source name, but only the dates of examination regulation or effective from metadata.
        1. Best Fit: Select the source where the study program AND degree AND (examination regulation OR effective from) match the METADATA.
        2. Fallback Fit: If no source satisfies Rule 1, select the source where the study program AND degree match the METADATA. Among these, choose the one where the examination regulation or effective from is the closest occurring chronologically BEFORE the METADATA.
        3. No Fit: If no source satisfies either Rule 1 or Rule 2 (i.e., no match for both study program and degree), return "unknown".

        Important: Return ONLY the source of the module handbook as a single line of text, with no explanation, implementation, or additional information.

        METADATA:
        {metadata_string}

        LIST:
        {sources_string}
    """

    llm = ChatOpenAI(model_name=model, temperature=0)

    response = llm.invoke(prompt)
    source = response.content.strip()

    # check if filepath of source exists
    # TODO: change into a more general way to check if the source exists
    if not os.path.exists(source):
        if os.path.exists(f"module_handbooks_economics/{source}"):
            source = f"module_handbooks_economics/{source}"
        else:
            source = "unknown"

    return source

def build_vectorstore(metadata_text_file="module_handbooks_economics_metadata.txt", persist_directory="vectorstores", collection_name="sources_metadata"):
    """
    Embed sources and save them in a vectorstore.
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model)

    # Read the sources from the metadata file
    with open(metadata_text_file, "r", encoding="utf-8") as f:
        sources = [eval(line.strip()) for line in f.readlines()]  # Each line is a dictionary

    # Create the documents for the vectorstore with the source as metadata
    documents = []
    for source in sources:
        source_str = f"{source['study_program']} {source['degree']} {source['effective_from']} {source['examination_regulation']}"
        documents.append(
            Document(
                page_content=source_str,
                metadata={"source": source["source"]}
            )
        )
    
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    ) 
    print("Saved to vectorstore.")

def identify_source_retriever(extracted_metadata, persist_directory="vectorstores", collection_name="sources_metadata"):
    """
    Identify the source of the module handbook using a vectorstore.
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    metadata_str = f"{extracted_metadata['study_program']} {extracted_metadata['degree']} {extracted_metadata['effective_from']} {extracted_metadata['examination_regulation']}"

    result = vectorstore.similarity_search_with_score(metadata_str, k=1)
    score = result[0][1]
    source = result[0][0].metadata["source"]

    return source


def identify_module_title(source, extracted_title):
    """
    Identify the module title based on the source and the extracted module title.
    """
    if source == "unknown" or extracted_title == "unknown":
        return "unknown"

    # Similarity search with all modules of the source
    # First, extract all module titles from the source -> load all chunks of the source and extract the module titles
    with open("chunks_unmodified.pkl", "rb") as f:
        chunks = pickle.load(f)

    chunks_of_source = [chunk for chunk in chunks if chunk.metadata["source"] == source]
    module_titles = []
    for chunk in chunks_of_source:
        title = chunk.metadata["module_title"]
        if title not in module_titles:
            module_titles.append(title)

    # Then, calculate the similarity between the extracted module title and all module titles of the source
    similarity_scores = []
    for title in module_titles:
        similarity_score = SequenceMatcher(None, extracted_title, title).ratio()
        similarity_scores.append(similarity_score)

    # return the module title with the highest similarity score (but it should be above a certain threshold)
    max_similarity_score = max(similarity_scores)
    if max_similarity_score > 0.5:
        return module_titles[similarity_scores.index(max_similarity_score)]
    else:
        return "unknown"


def generate_filter(question):
    filter = {}

    extracted_metadata = extract_metadata(question)
    source = identify_source(extracted_metadata) # or identify_source_retriever(extracted_metadata)
    module_title = identify_module_title(source, extracted_metadata["module_title"])

    if source != "unknown" and module_title != "unknown":
        filter = {
            "$and": [
                {"source": {"$eq": source}},
                {"module_title": {"$eq": module_title}},
            ]
        }

    elif source != "unknown":
        filter = {"source": {"$eq": source}}

    return filter


def format_splits(relevant_splits):
    """
    Join the relevant splits from the retriever.
    """
    return "\n\n".join(
        relevant_split.page_content for relevant_split in relevant_splits
    )


def rag_chain(question, filter):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model)

    # Load vectorstore with the unmodified chunks
    persist_directory = "vectorstores"
    collection_name = "module_handbooks_economics_unmodified_chunks"
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # Set up retriever with the metadata filter
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "filter": filter})

    # Define prompt template
    template = """Answer the question based only on the following context. Don't mention the word 'context' in your answer, rather use 'my information' or similar. 
    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize LLM
    llm = ChatOpenAI(model_name=model, temperature=0)

    # Build the chain
    rag_chain = (
        {"context": retriever | format_splits, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    return answer


def rag_system(question):
    filter = generate_filter(question)
    answer = rag_chain(question, filter)
    return answer


def save_all_sources():
    """
    Save the metadata of all module handbooks in a text file.
    """
    with open("module_handbooks_economics.pkl", "rb") as f:
        module_handbooks = pickle.load(f)

    for module_handbook in module_handbooks:
        # string = f"Study Program: {module_handbook.metadata["study_program"]}, Degree: {module_handbook.metadata["degree"]}, Effective From: {module_handbook.metadata["effective_from"]}, Examination Regulation: {module_handbook.metadata["examination_regulation"]}, Source: {module_handbook.metadata["source"]}"
        with open("module_handbooks_economics_metadata.txt", "a") as f:
            f.write(f"{module_handbook.metadata} \n")


def main():
    # Example questions are extended by all necessary information to identify the right source (as used in the manual metadata filtering to ensure the same results)
    examples = [
        {
            "question": "Welche Prüfungsleistungen muss ich für Digital Business absolvieren? Und wie viele Leistungspunkte gibt das Modul? Ich habe den Wirtschaftsinformatik Bachelor im Wintersemester 2019/20 angefangen.",
            "expected_answer": "Veranstaltungsbegleitende Gruppenarbeiten, Schriftliche Abschlussprüfung (Nr. 1), 50%, 50%",
        },
        # {
        #     "question": "Wie lang muss meine Bachelorarbeit sein? Und wie viel macht die Bachelorarbeitsnote von meinem Gesamtdurchnitt aus? Ich habe den Wirtschaftsinformatik Bachelor im Wintersemester 2019/20 angefangen.",
        #     "expected_answer": "40 Seiten, 6,78%",
        # }
    ]

    # Uncomment the following lines when running the first time to save all sources in a text file and (optionally) build the vectorstore
    # save_all_sources()
    # build_vectorstore()

    for example in examples:
        question = example["question"]
        expected_answer = example["expected_answer"]

        answer = rag_system(question)
        print(f"{question} \n\n{answer} \n\nExpected answer: {expected_answer}")
        print("\n" + "-" * 100 + "\n")


if __name__ == "__main__":
    main()
