from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import(
    create_stuff_documents_chain
)
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import sys

load_dotenv()

# **********Write the code from here onwards:


#***************models******************

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

model = ChatGroq(
      model = "openai/gpt-oss-20b"
)

#****************************************


video_url = input("Enter Youtube video URL:").strip()

loader = YoutubeLoader.from_youtube_url(
    video_url,
    add_video_info=False,
    language=["en","en-US"],
)


transcript = loader.load()
transcript = transcript[0].page_content


splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.create_documents([transcript])


vector_store = FAISS.from_documents(chunks,embedding_model)

retriever = vector_store.as_retriever(search_type="similarity",
                                      search_kwargs={"k":4})

prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer Only from the provided transcript context.
        If the context is not present, just say you don't know.
        {context}
        Question:{question}
    """,
    input_variables = ['context','question']
)

question = input("Enter the question?")

retrieved_docs    = retriever.invoke(question) 

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text,"question":question})

answer = model.invoke(final_prompt)

print(answer.content)
