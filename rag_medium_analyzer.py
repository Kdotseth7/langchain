from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langchain.chains import VectorDBQA
from langchain_openai import OpenAI

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    
    loader = TextLoader("./documents/medium_blog.txt")
    document = loader.load()
    # print(document)
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, 
                                          chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))
    
    embeddings = OpenAIEmbeddings()
    
    index_name="medium-blogs-embedding-index"
    DIMENSIONS = 1536
    docsearch = Pinecone.from_documents(texts, 
                                        embeddings, 
                                        index_name=index_name)
    
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), 
                                    chain_type="stuff", 
                                    vectorstore=docsearch, 
                                    return_source_documents=True)
    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    response = result["result"]
    print(response)