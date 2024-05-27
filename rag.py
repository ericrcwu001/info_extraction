#!/usr/bin/python3

from os.path import splitext, join, exists
from shutil import rmtree
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from models import Llama3
from prompts import rag_template

class RAG(object):
  def __init__(self, file_path, db_dir = 'db', locally = False):
    if exists(db_dir): rmtree(db_dir)
    # load llama3
    tokenizer, llm = Llama3(locally)
    # load pdf to vectordb
    docs = list()
    stem, ext = splitext(file_path)
    if ext.lower() == '.pdf':
      loader = UnstructuredPDFLoader(file_path, mode = "single", strategy = "fast")
    elif ext.lower() == '.txt':
      loader = TextLoader(file_path)
    elif ext.lower() in ['.html', '.htm']:
      loader = UnstructuredHTMLLoader(file_path)
    else:
      raise Exception('unknown file format!')
    docs.extend(loader.load())
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
    self.chain = RetrievalQA.from_chain_type(llm, retriever = vectordb.as_retriever(), return_source_documents = True, chain_type_kwargs = {"prompt": prompt})
  def query(self, question):
    res = self.chain({'query': question})
    return res['result'].replace('assistant\n\n',''), res['source_documents']

if __name__ == "__main__":
  rag = RAG('CN117175037 固态电解质浆料、固态电解质膜、固态电池及用电装置.html', locally = True)
  res, support = rag.query("负极材料属于['碳基','硅基','锂金属或锂合金','氧化物','硫化物']中的哪一类？如果材料不在这些类别中，给出具体材料名称。")
  print('负极：',res, support)
  res, support = rag.query("正极材料属于['氧化钴锂','氧化镍锂','氧化锰锂','磷酸亚铁锂','硫化物']中的哪一类？如果材料不在这些类别中，给出具体材料名称。")
  print('正极：',res, support)
  res, support = rag.query("电解质属于['聚合物','氧化物','硫化物']中的哪一类？如果电解质不在这些类别中，给出具体电解质名称。")
  print('电解质：',res, support)
  res, support = rag.query("电池结构属于['wound cell','stacked cell']中的哪一类？如果电池结构不再这些类别中，给出具体电池结构名称。")
  print('电池结构：',res, support)
  res, support = rag.query("公司名字是什么？")
  print("公司名字：",res, support)
