from langchain_google_genai import ChatGoogleGenerativeAI ,GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from youtube_transcript_api import YouTubeTranscriptApi ,TranscriptsDisabled
import os 
os.environ['GOOGLE_API_KEY'] = 'AIzaSyB8du31T_IfbT5L61IjOr8T2HpgaINNk7I'

#STEP 1: DOCUMENT INGESTION:

video_id = 'wjZofJX0v4M'
try:
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_id , languages=['en'])
    
    transcript_list = ' '.join(chunks.text for chunks in  transcript)
    print(transcript)

except  TranscriptsDisabled():
    print('No transcript found')
except Exception as e:
  print(f"An error occurred: {e}")


splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap =200)
chunks = splitter.create_documents([transcript])

embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
vectorstore = Chroma.from_documents(chunks , embeddings)

retriever = vectorstore.as_retriever(search_type = 'mmr' , search_kwargs ={'k':6 , 'lambda_mult' : 0.25})
#retriever

result = retriever.invoke('what are llm?')
print(result[0].page_content)


#STEP 2 : DATA AUGMENTATION--

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',
                               temperature= 0.2 , 
                               google_api_key = 'AIzaSyB8du31T_IfbT5L61IjOr8T2HpgaINNk7I')

prompt = PromptTemplate(
        template= ''' You are a helpful ai assistance answer the following question  from the transcript {context} given context given.If you dont knoe the context just say i dont know
        conetext : {context} 
        Question : {question}''',
        input_variables=['context' , 'question']
)

question = 'what is DL?'
retrieval_docs = retriever.invoke(question)

context_text = ' '.join(docs.page_content for docs in retrieval_docs)
context_text

final_prompt = prompt.invoke({'context' : context_text , 'question' : question})
final_prompt


#STEP 3: GENERATION--


response = model.invoke(final_prompt)
print(response)