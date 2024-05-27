#!/usr/bin/python3

from os.path import splitext, join
from shutil import rmtree
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from models import Llama3
from prompts import rag_template

class RAG(object):
  def __init__(self, pdf_path, db_dir = 'db', locally = False):
    if exists(db_dir): rmtree(db_dir)
    # load llama3
    tokenizer, llm = Llama3(locally)
    # load pdf to vectordb
    docs = list()
    loader = UnstructuredPDFLoader(pdf_path, mode = "single", strategy = "fast")
    docs.append(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
    split_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Chroma.from_documents(
        documents = split_docs,
        embedding = embeddings,
        persist_directory = db_dir)
    vectordb.persist()
    # create chain
    prompt = rag_template(tokenizer, False)
    self.chain = RetrievalQA.from_chain_type(llm, retriever = db.as_retriever(), return_source_documents = True, chain_type_kwargs = {"prompt": prompt})
  def query(self, question):
    return self.chain({'query': question})

