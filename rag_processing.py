import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List
from io import BytesIO
import re
import tempfile
from dotenv import load_dotenv
import os

try :
    os.environ["GOOGLE_API_KEY"] = st.secrets['gemini_api_key']
    
except Exception as e:
    load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-2.5-pro" if supported in your setup
    temperature=0.5,
    max_retries=2
)


prompt = PromptTemplate(input_variables=["context","question"],
                        template="""You are an AI career assistant specialized in analyzing resumes.

                        Use the following extracted parts of the resume as context:
                        {context}

                        Answer the user's question in a helpful and professional way.
                        If the answer is not directly available in the resume, say "The information is not present in the resume."
                        Do not make up any details.

                        Question: {question}
                        Answer:""")



jd_prompt = PromptTemplate(input_variables=["resume_text","job_desc"],
                        template="""
                        You are an expert career consultant and professional writer.  
                        Create a compelling, personalized cover letter for a job application.  

                        Candidate Resume:
                        {resume_text}

                        Job Description:
                        {job_desc}

                        Guidelines:
                        - Professional and concise tone
                        - Tailor it to the job role
                        - Highlight relevant experience, skills, and achievements
                        - Keep within 350 words, 3–4 short paragraphs
                        - End with a polite and confident closing
                        - Optimize for ATS readability
                        
                        """)

class ATSResult(BaseModel):
    score: float
    matched_skills: List[str]
    missing_skills: List[str]

ats_parser = PydanticOutputParser( pydantic_object = ATSResult)
ats_prompt = PromptTemplate(input_variables=["resume_text","job_desc"],
                        template="""
                        You are an ATS scoring assistant.

                            Job Description:
                            {job_desc}

                            Resume:
                            {resume_text}

                            Task:
                            1. Identify key skills and qualifications in the JD.
                            2. Compare with the resume content.
                            /n {format_instruction}
                        
                        """,partial_variables={"format_instruction":ats_parser.get_format_instructions()})

                    



parser = StrOutputParser()

def markdown_to_reportlab(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"_(.*?)_", r"<i>\1</i>", text)
    text = text.replace("\n\n", "<br/><br/>")  # Paragraph breaks
    return text


def load_and_split_store(file: str):
    """
    Loads and splits documents into chunks for RAG.
    
    Args:
        file_path (str): Path to the file (PDF/DOCX/TXT).
       
    
    Returns:
        List[Document]: Chunks of the document with metadata.
    """
    global docs
    ext = os.path.splitext(file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    # Choose loader based on file extension
    
    if ext == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(tmp_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(tmp_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Load document
    docs = loader.load()
    
    # Different splitting strategy for Resume vs JD
   
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    splitted_text = splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    chrome = Chroma()
    vector_store = chrome.from_documents(documents=splitted_text,
                        embedding=embedding_model,
                        collection_name="my_collection")
    base_retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3})

    return base_retriever


def get_chat_details(retriever,query):
    rag_context = RunnableParallel( {"context": retriever | RunnableLambda(lambda x:" ".join([j.page_content for j in x])),
                                "question":RunnablePassthrough()})
                        
    chain_1 = rag_context| prompt | llm | parser
    llm_response = chain_1.invoke(query)
    return llm_response
    
def get_cover_letter(job_desc):
    query = {"resume_text": docs,"job_desc":job_desc}
                        
    chain_1 = jd_prompt | llm | parser
    llm_response = chain_1.invoke(query)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    formatted_text = markdown_to_reportlab(llm_response)
    story.append(Paragraph(formatted_text, styles["Normal"]))
    story.append(Spacer(1, 12))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()


    return llm_response,pdf_bytes


def get_ats_detials(jd_text):


    query = {"resume_text": docs,"job_desc":jd_text}

    chain_1 = ats_prompt | llm | ats_parser
    llm_response = chain_1.invoke(query)

    return llm_response