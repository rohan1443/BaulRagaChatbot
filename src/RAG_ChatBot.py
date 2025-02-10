# RAG_ChatBot.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
import os

load_dotenv()

class ChatBot:
    def __init__(self, embeddings, index, index_name):
        self.embeddings = embeddings
        self.index = index
        self.index_name = index_name

        self.docsearch = PineconeVectorStore(index, embeddings, index_name=index_name)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")  # Or gpt-4 if available
        self.prompt = PromptTemplate(
            template="""
            You are a chatbot Bengal travel assistant who only speak about the culture and details of the Bauls of Bengal. Users will ask you questions about their culture, people, folk songs, instruments. Use the following piece of context to answer the question.
            If you don't know the answer, just say you don't know.
            Your answer should be short and concise, no longer than 2 sentences.

            Context: {context}
            Question: {question}
            Answer:
            """,
            input_variables=["context", "question"],
        )
        self.rag_chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.docsearch.as_retriever(), chain_type_kwargs={"prompt": self.prompt}
        )

        # Load and split documents (This is done ONCE during ChatBot instantiation)
        pdf_path = os.path.join(os.getcwd(), "materials", "The_Bauls_of_Bengal.pdf")
        loader = PyPDFLoader(pdf_path)  # Correct path!
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Store documents in Pinecone (This is done ONCE during ChatBot instantiation)
        self.docsearch.add_documents(docs) # Use add_documents for efficiency


    def query(self, question):
        return self.rag_chain({"query": question})["result"]