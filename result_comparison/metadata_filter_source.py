import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

###### RAG-SYSTEM FOR MODULE HANDBOOKS - with manual metadata  (but don't include module in filter) ######
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

examples = [
    {
        "question": "Was sind Module im 1. Semester Bachelor Wirtschaftsinformatik in der PO 2022?",
        "expected_answer": "Einführung in die Wirtschaftsinformatik, Programmierung, Wirtschaftsmathematik, Einführung in die BWL, Investition und Finanzierung",
        "filter": {
            "source": {
                "$eq": "module_handbooks_economics/2024-01-11_bsc_wi_modulbeschreibungen.pdf"
            }
        },
    },
    {
        "question": "Was für Module sind im Bereich Quantitative Methoden im Bachelor Wirtschaftsinformatik in der PO 2018?",
        "expected_answer": "Wirtschaftsmathematik, Operations Research, Daten und Wahrscheinlichkeiten, Datenanalyse und Simulation",
        "filter": {
            "source": {
                "$eq": "module_handbooks_economics/2018-10-05_bsc_wi_modulbeschreibungen.pdf"
            }
        },
    },
    {
        "question": "Was für Pflichtmodule muss ich im Bereich VWL und Recht im Studiengang Bachelor BWL belegen?",
        "expected_answer": "Einführung in die VWL & Mikroökonomik, Grundlagen der Makroökonomik, Recht für Ökonomen",
        "filter": {
            "source": {
                "$eq": "module_handbooks_economics/2024-01-23_bsc_bwl_modulhandbuch.pdf"
            }
        },
    },
    {
        "question": "What are the prerequisites to write the master's thesis in Information Systems?",
        "expected_answer": "60 credit points",
        "filter": {
            "source": {
                "$eq": "module_handbooks_economics/2022-09-23_msc_is_module_compendium.pdf"
            }
        },
    },
    {
        "question": "Worum geht es in dem Modul 'Regulierungsökonomik' des Studiengangs Master VWL?",
        "expected_answer": "In diesem Modul wird die Ursachenanalyse für Marktversagen vertieft und das ökonomische Instrumentarium zu deren Korrektur und Regulierung untersucht, …",
        "filter": {
            "source": {
                "$eq": "module_handbooks_economics/2020-04-27_msc_ec_modulbeschreibungen.pdf"
            }
        },
    },
    {
        "question": "In welchem Semester findet die Vorlesung 'Grundlagen der Regulierung' statt?",
        "expected_answer": "Jedes Sommersemester -> Studiengang Bachelor VWL ab WS 2020/21",
        "filter": {
            "source": {
                "$eq": "module_handbooks_economics/2020-06-16_bsc_vwl_modulbeschreibungen.pdf"
            }
        },
    },
    {
        "question": "Welchen Umfang muss die Bachelorarbeit im Studiengang BWL haben?",
        "expected_answer": "Nicht mehr als 7000 Wörter",
        "filter": {
            "source": {
                "$eq": "module_handbooks_economics/2022-12-08_bsc_bwl_modulhandbuch.pdf"
            }
        },
    },
    {
        "question": "Wie setzt sich die Note aus dem Modul 'Performance Management & Strategy Execution' des Masters BWL ab WiSe 2023/24 zusammen?",
        "expected_answer": "Klausur 75%, Schriftliche Ausarbeitung 25%",
        "filter": {
            "source": {
                "$eq": "module_handbooks_economics/2024-03-05_msc_bwl_modulhandbuch.pdf"
            }
        },
    },
    {
        "question": "In welchem Semester ist der Praxisworkshop im Master BWL mit Major Finance vorgesehen?",
        "expected_answer": "Im 3. Fachsemester",
        "filter": {
            "source": {
                "$eq": "module_handbooks_economics/2024-03-05_msc_bwl_modulhandbuch.pdf"
            }
        },
    },
]

i = 8  #### go through each example manually here (run 9 times) ####
result_file = (
    f"result_comparison/metadata_filter_source_retriever_chunks/question{i+1}.txt"
)
filter = examples[i]["filter"]
question = examples[i]["question"]
expected_answer = examples[i]["expected_answer"]


def format_splits(relevant_splits):
    """
    Join the relevant splits from the retriever.
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
    with open(result_file, "w") as f:
        f.write(f"Metadata filter (only source) - Question: {question} \n\n")  ########
        f.write("===== Metadata =====\n\n")
        f.write(all_metadata_str)
        f.write("\n\n===== Full Splits =====\n\n")
        f.write(all_splits_str)

    return "\n\n".join(
        relevant_split.page_content for relevant_split in relevant_splits
    )


def main():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model)

    persist_directory = "vectorstores"
    collection_name = "module_handbooks_economics_unmodified_chunks"
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # Set up retriever
    # k=10 so that there is as much context/information as possible, but still enough input-tokens for the question
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

    with open(result_file, "a") as f:  # Open in append mode
        f.write("\n\n===== Answer =====\n\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"Expected answer: {expected_answer}\n\n")
        f.write(f"Filter: {filter}\n")

    print(f"{question} \n\n{answer} \n\nExpected answer: {expected_answer}")
    print("\n" + "-" * 100 + "\n")


if __name__ == "__main__":
    main()
