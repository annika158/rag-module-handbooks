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
import chromadb


###### RAG-SYSTEM FOR MODULE HANDBOOKS - with metadata filtering implemented ######
# Prerequisite: module handbooks are already processed, chunked and saved in the vectorstore (see main.py)

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoids parallelism error in embedding

# Set API key and models
api_key = os.getenv("OPENAI_API_KEY")
base_url = "https://gpt.uni-muenster.de/v1"
model = "Llama-3.1-70B"  # Options: "Llama-3.1-70B", "mixtral-8x7B"
huggingface_model = "sentence-transformers/all-mpnet-base-v2"  # example alternative: "sentence-transformers/all-MiniLM-L6-v2"

os.environ["OPENAI_API_BASE"] = base_url
os.environ["OPENAI_API_KEY"] = api_key


i = 0 #### go through each example manually here (run 9 times) ####
result_file = f"result_comparison/rag_metadata_filter_source_retrieved_chunks/question{i+1}.txt"

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
        ResponseSchema(
            name="module_title",
            description="Name or title of the Module asked for (if multiple, choose only one). Examples include 'Bachelor's thesis', 'Operations Research', or other specific academic modules.",
        ),
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
        print(f"Error processing user query: {e}")
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
    # it is not guaranteed that the right source is identified but for the modified examples it works
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


def build_vectorstore(
    metadata_text_file="module_handbooks_economics_metadata.txt",
    persist_directory="vectorstores",
    collection_name="sources_metadata",
):
    """
    Embed sources and save them in a vectorstore.
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model)

    # Read the sources from the metadata file
    with open(metadata_text_file, "r", encoding="utf-8") as f:
        sources = [
            eval(line.strip()) for line in f.readlines()
        ]  # Each line is a dictionary

    # Create the documents for the vectorstore with the source as metadata
    documents = []
    for source in sources:
        source_str = f"{source['study_program']} {source['degree']} {source['effective_from']} {source['examination_regulation']}"
        documents.append(
            Document(page_content=source_str, metadata={"source": source["source"]})
        )

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print("Saved to vectorstore.")


def identify_source_retriever(
    extracted_metadata,
    persist_directory="vectorstores",
    collection_name="sources_metadata",
):
    """
    Identify the source of the module handbook using a vectorstore.
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    metadata_str = f"{extracted_metadata['study_program']} {extracted_metadata['degree']} {extracted_metadata['effective_from']} {extracted_metadata['examination_regulation']}"

    result = vectorstore.similarity_search_with_score(metadata_str, k=1)
    score = result[0][1]
    source = result[0][0].metadata["source"]

    return source


def generate_filter(question):
    filter = {}

    extracted_metadata = extract_metadata(question)
    source = identify_source(
        extracted_metadata
    )  # or identify_source_retriever(extracted_metadata)

    if source != "unknown":
        filter = {"source": {"$eq": source}}
    
    with open(result_file, "w") as f:  # Open in append mode
        f.write(f"Metadata filter (only source): \n\n")
        f.write(f"Filter:\n {filter}\n\n")

    return filter


def save_retrieved_chunks(relevant_splits):
    """
    Save the retrieved chunks to a text file.
    """
    all_metadata_str = ""
    all_splits_str = ""
    # Save the relevant splits as a one text file (first all metadata, then the whole split)
    for idx in range(len(relevant_splits)):
        all_metadata_str += (
            "-" * 10 + f" Chunk {idx} " + "-" * 10 + "\n\n"
            f"Source: {relevant_splits[idx].metadata['source']}\n"
            f"Study Program: {relevant_splits[idx].metadata['study_program']}\n"
            f"Degree: {relevant_splits[idx].metadata['degree']}\n"
            f"Examination Regulation: {relevant_splits[idx].metadata['examination_regulation']}\n"
            f"Effective From: {relevant_splits[idx].metadata['effective_from']}\n"
            f"Module Title: {relevant_splits[idx].metadata['module_title']}\n"
            "\n"
        )
        all_splits_str += (
            "-" * 10 + f" Chunk {idx} " + "-" * 10 + "\n"
            f"{relevant_splits[idx].metadata}\n"
            f"{relevant_splits[idx].page_content}\n"
        )

    # Save the relevant splits as a text file - first all_metadata, then all_splits
    with open(result_file, "a") as f:
        f.write("===== Metadata =====\n\n")
        f.write(all_metadata_str)
        f.write("\n\n===== Full Splits =====\n\n")
        f.write(all_splits_str)


def format_splits(relevant_splits):
    """
    Join the relevant splits from the retriever.
    """
    save_retrieved_chunks(relevant_splits)
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
            "question": "Was sind Module im 1. Semester Bachelor Wirtschaftsinformatik in der PO 2022?",
            "expected_answer": "Einführung in die Wirtschaftsinformatik, Programmierung, Wirtschaftsmathematik, Einführung in die BWL, Investition und Finanzierung",
        },
        {
            "question": "Was für Module sind im Bereich Quantitative Methoden im Bachelor Wirtschaftsinformatik in der PO 2018?",
            "expected_answer": "Wirtschaftsmathematik, Operations Research, Daten und Wahrscheinlichkeiten, Datenanalyse und Simulation",
        },
        {
            "question": "Was für Pflichtmodule muss ich im Bereich VWL und Recht im Studiengang Bachelor BWL belegen? Ich habe im Wintersemester 2023/24 angefangen.",
            "expected_answer": "Einführung in die VWL & Mikroökonomik, Grundlagen der Makroökonomik, Recht für Ökonomen",
        },
        {
            "question": "What are the prerequisites to write the master's thesis in Information Systems? I started in Wintersemester 2019/20.",
            "expected_answer": "60 credit points",
        },
        {
            "question": "Worum geht es in dem Modul 'Regulierungsökonomik' des Studiengangs Master VWL? Ich habe im WS 2020/21 angefangen.",
            "expected_answer": "In diesem Modul wird die Ursachenanalyse für Marktversagen vertieft und das ökonomische Instrumentarium zu deren Korrektur und Regulierung untersucht, …",
        },
        {
            "question": "In welchem Semester findet die Vorlesung 'Grundlagen der Regulierung' statt? Ich studiere VWL im Bachelor seit dem WS 2020/21.",
            "expected_answer": "Jedes Sommersemester -> Studiengang Bachelor VWL ab WS 2020/21",
        },
        {
            "question": "Welchen Umfang muss die Bachelorarbeit im Studiengang BWL mit der PO 2022 haben?",
            "expected_answer": "Nicht mehr als 7000 Wörter",
        },
        {
            "question": "Wie setzt sich die Note aus dem Modul 'Performance Management & Strategy Execution' des Masters BWL ab WiSe 2023/24 zusammen?",
            "expected_answer": "Klausur 75%, Schriftliche Ausarbeitung 25%",
        },
        {
            "question": "In welchem Semester ist der Praxisworkshop im Master BWL vorgesehen? Ich fange im Wintersemester 2024/25 an.",
            "expected_answer": "Im 3. Fachsemester",
        },
    ]

    # Uncomment the following lines when running the first time to save all sources in a text file and (optionally) build the vectorstore
    # save_all_sources()
    # build_vectorstore()

    
    question = examples[i]["question"]
    expected_answer = examples[i]["expected_answer"]

    answer = rag_system(question)
    print(f"{question} \n\n{answer} \n\nExpected answer: {expected_answer}")
    print("\n" + "-" * 100 + "\n")

    with open(result_file, "a") as f:  # Open in append mode
        f.write("\n\n===== Answer =====\n\n")
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"Expected answer: {expected_answer}\n")


if __name__ == "__main__":
    main()
