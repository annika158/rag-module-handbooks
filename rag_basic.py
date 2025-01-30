import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

###### SIMPLE RAG-SYSTEM FOR MODULE HANDBOOKS ######
# Prerequisite: module handbooks are already processed, chunked and saved in the vectorstore (see preparation.py)

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoids parallelism error in embedding

# Set API key and models
api_key = os.getenv("OPENAI_API_KEY")
base_url = "https://gpt.uni-muenster.de/v1"
model = "Llama-3.1-70B"  # Options: "Llama-3.1-70B", "mixtral-8x7B"
huggingface_model = "sentence-transformers/all-mpnet-base-v2"  # needs to be the same as used for embedding the module handbooks in preparation.py!

# TODO: Change the names of the directory and collection according to your local setup
persist_directory = "vectorstores"
collection_name = "module_handbooks_economics_unmodified_chunks"

os.environ["OPENAI_API_BASE"] = base_url
os.environ["OPENAI_API_KEY"] = api_key

def format_splits(relevant_splits):
    """
    Join the relevant splits from the retriever.
    """
    # To look at the retrieved splits, print here:
    # for relevant_split in relevant_splits:
    #     print(relevant_split)
    #     print("\n\n" + "-"*100 + "\n\n")
    return "\n\n".join(relevant_split.page_content for relevant_split in relevant_splits)

def main():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model)

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

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
        "question": "Was für Pflichtmodule muss ich im Bereich VWL und Recht im Studiengang Bachelor BWL belegen?",
        "expected_answer": "Einführung in die VWL & Mikroökonomik, Grundlagen der Makroökonomik, Recht für Ökonomen",
    },
    {
        "question": "What are the prerequisites to write the master's thesis in Information Systems?",
        "expected_answer": "60 credit points",
    },
    {
        "question": "Worum geht es in dem Modul 'Regulierungsökonomik' des Studiengangs Master VWL?",
        "expected_answer": "In diesem Modul wird die Ursachenanalyse für Marktversagen vertieft und das ökonomische Instrumentarium zu deren Korrektur und Regulierung untersucht, …",
    },
    {
        "question": "In welchem Semester findet die Vorlesung 'Grundlagen der Regulierung' statt?",
        "expected_answer": "Jedes Sommersemester -> Studiengang Bachelor VWL ab WS 2020/21",
    },
    {
        "question": "Welchen Umfang muss die Bachelorarbeit im Studiengang BWL haben?",
        "expected_answer": "Nicht mehr als 7000 Wörter",
    },
    {
        "question": "Wie setzt sich die Note aus dem Modul 'Performance Management & Strategy Execution' des Masters BWL ab WiSe 2023/24 zusammen?",
        "expected_answer": "Klausur 75%, Schriftliche Ausarbeitung 25%",
    },
    {
        "question": "In welchem Semester ist der Praxisworkshop im Master BWL mit Major Finance vorgesehen?",
        "expected_answer": "Im 3. Fachsemester",
    },
    ]

    for example in examples:
        question = example["question"]
        expected_answer = example["expected_answer"]

        answer = rag_chain.invoke(question)
        print(f"{question} \n\n{answer} \n\nExpected answer: {expected_answer}")
        print("\n" + "-" * 100 + "\n")

if __name__ == "__main__":
    main()