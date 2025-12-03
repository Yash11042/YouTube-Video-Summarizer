from langchain_google_genai import ChatGoogleGenerativeAI ,GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.output_parsers import StructuredOutputParser
from langchain_core.runnables import RunnableParallel , RunnablePassthrough , RunnableSequence , RunnableLambda

from youtube_transcript_api import YouTubeTranscriptApi ,TranscriptsDisabled
import os 
os.environ['GOOGLE_API_KEY'] = 'AIzaSyB8du31T_IfbT5L61IjOr8T2HpgaINNk7I'

video_id = 'wjZofJX0v4M'
try:
    
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_id , languages=['en'])
    
    transcript_list = ' '.join(chunks.text for chunks in  transcript)
    print(transcript)
except TranscriptsDisabled:
    print('No captional are available')

splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , 
                                          chunk_overlap = 200)
chunks = splitter.create_documents([transcript])


embedding = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
vectorstore = Chroma.from_documents(chunks , embedding)

retriever = vectorstore.as_retriever(search_type = 'mmr' , 
                                     search_kwargs = {'k': 6 , 'lambda_mult' : 0.25})


model = ChatGoogleGenerativeAI(model='gemini-1.5-flash' ,
                               temperature = 0.2,
                               google_api_key = 'AIzaSyB8du31T_IfbT5L61IjOr8T2HpgaINNk7I')

prompt = PromptTemplate(
                template= ''' You are a helpful ai assistance  answer the question in transcript {context} , if you dont know just say i dont know
                context : {context}
                question : {question}''',
                input_variables= ['context' , 'question'] 
)

#def format_text(retrieved_docs):
    #context_text = '\n\n'.join(docs.text for docs in retrieved_docs)
    #return context_text

format_docs = RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs))

parser = StructuredOutputParser()

parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs),
    'question' : RunnablePassthrough()
    })

final_chain = parallel_chain | prompt | model | parser

response = final_chain.invoke('what is LLM?')

print(response)

