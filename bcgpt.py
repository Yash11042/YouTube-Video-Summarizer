from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import os 

os.environ['GOOGLE_API_KEY'] = 'AIzaSyB8du31T_IfbT5L61IjOr8T2HpgaINNk7I'

video_id = 'wjZofJX0v4M'
try:
    # Correct method name is get_transcript() (lowercase 't')
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript_text = ' '.join([item['text'] for item in transcript])
except TranscriptsDisabled:
    print('No captions are available')
    transcript_text = ""  # Provide a fallback empty string

# Create documents from the transcript text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
chunks = splitter.split_text(transcript_text)
documents = [Document(page_content=chunk) for chunk in chunks]

embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
vectorstore = Chroma.from_documents(documents, embedding)

retriever = vectorstore.as_retriever(search_type='mmr', 
                                   search_kwargs={'k': 6, 'lambda_mult': 0.25})

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                             temperature=0.2)

prompt = PromptTemplate(
    template='''You are a helpful AI assistant. Answer the question based on the transcript context: {context}
    If you don't know the answer, just say "I don't know".
    
    Question: {question}''',
    input_variables=['context', 'question']
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the chain
chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | model
    | StrOutputParser()
)

response = chain.invoke("what is LLM?")
print(response)