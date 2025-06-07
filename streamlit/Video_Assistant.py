
import os
import streamlit as st
from langchain_openai import OpenAI, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from langchain.schema import Document
import re
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container

# Set up Streamlit page configuration
st.set_page_config(page_title=None,
                   page_icon=":cyclone:",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

# Access open AI key
OPENAI_API_KEY = st.secrets["YOUTUBE_OPENAI_API_KEY"]
openai_embed_model = "text-embedding-ada-002"
openai_model = "gpt-4o-mini"
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=openai_model, temperature=0.1)

# Styles Setup  #######################################################################

# Define header size/color:

def header():
    colored_header(
        label ="YouTube Chat Assistant",
        description = "Find a YouTube video with accurate captions. Enter url below.",
        color_name='blue-80'   
    )
    # additional styling
    st.markdown("""
        <style>
        /* Adjust the font size of the header */
        .st-emotion-cache-10trblm.e1nzilvr1 {
            font-size: 50px !important; /* Change this value to increase or decrease font size
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        /* Adjust the thickness of the line */
        hr {
            height: 14px !important; /* Change this value to increase or decrease line thickness
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        /* Adjust the font size of the description */
        div[data-testid="stCaptionContainer"] p {
            font-size: 20px !important; /* Change this value to increase or decrease font size
        }
        </style>
    """, unsafe_allow_html=True)

# Define button style/formatting
    
def video_button():
    with stylable_container(
        key="video",
        css_styles="""
            button {
                background-color: #0068ca;
                color: #ffffff;
                border-radius: 20px;
                }
                """
    ) as container:
        return st.button("Submit video")

def question_button():
    with stylable_container(
        key="question",
        css_styles="""
            button {
                background-color: #0068ca;
                color: #ffffff;
                border-radius: 20px;
                }
                """
    ) as container:
        return st.button("Submit question")

def vid_clear_button():
    with stylable_container(
        key="clear",
        css_styles="""
            button {
                background-color: #0068ca;
                color: #ffffff;
                border-radius: 20px;
                }
                """
    ) as container:
        return st.button("Clear History")
    
# End styles section ###########################################################################

# Define functions ##########################################################################

# Define function to clear history
def clear_vid_chat_history():
    st.session_state['vid_chat_history'] = []

def question_button_and_style():
    submit_question = question_button()
    st.markdown("""
        <style>
        /* Adjust the font size of the input labels */
        .st-emotion-cache-ue6h4q p {
        font-size: 20px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    return submit_question

def display_response(response, vid_chat_history):
    st.write(response)
    st.divider()
    st.markdown(f"**Conversation History**")
    for prompts in reversed(vid_chat_history):
        st.markdown(f"**Question:** {prompts[0]}")
        st.markdown(f"**Answer:** {prompts[1]}")
        st.divider()  

def reset_session_state(keys=None):
    """ Reset specific keys in session state or all if keys is None """
    if keys is None:
        st.session_state.clear()
    else:
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]

def fetch_youtube_transcript(url, proxy_user, proxy_pass, proxy_host, proxy_port):
    # Extract the video ID from the URL
    match = re.search(r"(?:v=|be/)([\w-]+)", url)
    if not match:
        st.error("Invalid YouTube URL")
        return []
    video_id = match.group(1)

    # Set up the proxy string
    proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"

    # Pass proxies as dict
    proxies = {"https": proxy_url}

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies)
        text = "\n".join([seg['text'] for seg in transcript])
        doc = Document(page_content=text, metadata={'source': url})
        return [doc]
    except Exception as e:
        st.error(f"Transcript retrieval failed: {e}")
        return []
def process_video(url):
    # Fill in your proxy credentials (or pull from secrets for security!)
    proxy_user = st.secrets["PROXY_USER"]
    proxy_pass = st.secrets["PROXY_PASS"]
    proxy_host = st.secrets["PROXY_HOST"]
    proxy_port = st.secrets["PROXY_PORT"]

    documents = fetch_youtube_transcript(url, proxy_user, proxy_pass, proxy_host, proxy_port)
    if not documents:
        st.error("No captions found in video. Please try a different video.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=40000, chunk_overlap=600)
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        st.error("Failed to generate text chunks from the video captions.")
        return None
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    vector_store = FAISS.from_documents(chunks, embedding)
    return vector_store

# def process_video(url):
#     loader = YoutubeLoader.from_youtube_url(url)
#     documents = loader.load()
#     if not documents:
#         st.error("No captions found in video. Please try a different video.")
#         return None
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=40000, chunk_overlap=600)
#     chunks = text_splitter.split_documents(documents)
#     if not chunks:
#         st.error("Failed to generate text chunks from the video captions.")
#         return None
#     embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model = "text-embedding-ada-002")
#     vector_store = FAISS.from_documents(chunks, embedding)
#     return vector_store

def process_question(vector_store, question):
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm = llm, chain_type='stuff', retriever=retriever)
    results = qa.invoke(question)
    return results


# Define main function
def main():
    
    header()       
    tab1, tab2 = st.tabs(["The App", "Accounting Use Cases"])
    
    with tab2:
        st.subheader("proof of concept")
        st.write("""Accountants use videos for numerous scenarios: recorded zoom trainings, onboardings, walkthroughs, meetings. But information in videos is
                 hard to access for repeated viewings, especially if the user only needs to revisit part of the content. This app ingests the transcript generated
                 from the captions and can answer questions or transform the video content to a useable format.""")
        st.markdown("**Examples:**")
        st.write("""
   - Create SOP from training video. Enter training video and ask for step-by-step instructions. 
   \n- \tGenerate quick summary of meeting or webcast that couldn't attend, but need the highlights.
   - Walkthroughs with auditors. Record the session with captioning, then feed it through a chatbot assistant to document steps, or ask follow-up questions if details are forgotten.
   - Train the chatbot on a full repository of onboarding training videos for new hires.  When users ask questions, it provides reponses, links to specific videos, and links to other internal documents with further information.
   - Empower team members to self-serve support for systems/processes/policies.

  **Example queries:**
   - Give me a bulleted list of the main talking points with a summary of each.
   - Provide step-by-step instructions for the process described in the video.
   - At what points in the video is 'xyz topic' mentioned? 
   - What is the most important point of the video?


**Features:**

 - AI driven conversational interface
 - Uses gpt-4o-mini for generating responses
 - Embeddings are stored in vector store, enabling rapid searches

**Technologies used:**
 - **Streamlit**: To create the web interface and community cloud hosting
 - **LangChain**: Foundational framework connecting OpenAI's models and FAISS vector storage
 - **OpenAI's gpt-4o-mini**: For natural language understanding and response generation
 - **OpenAI's text-embedding-ada-002**: To create source content embeddings
 - **FAISS**: To store vectors and index them for fast retrieval""")
        
    with tab1:
            
        youtube_url = st.text_input('Input YouTube URL')
        submit_video = video_button()
        
        # Initialize vector_store and crc
        if 'vector_store' not in st.session_state:
            st.session_state['vector_store'] = []
        
        # initialize history
        if 'vid_chat_history' not in st.session_state:
            st.session_state['vid_chat_history'] = []
        
        # initialize question
        if 'question' not in st.session_state:
            st.session_state['question']  =[]
            
            
        if submit_video and youtube_url:
            with st.spinner("loading, chunking, and embedding..."):
                vector_store = process_video(youtube_url)
                if vector_store is None:
                    pass
                else:
                    st.session_state['vector_store'] = vector_store
                    st.success("Video processed and vector store created.")
            
        st.markdown("""
        <style>
        div[data-testid="InputInstructions"] > span:nth-child(1) {
            visibility: hidden;
        }
        </style>
        """, unsafe_allow_html=True)

        
        question = st.text_input("Ask me questions about the video!", placeholder="give me a bulleted list of the main talking points and a summary of each.")
        submit_question = question_button_and_style()
        # st.divider()
        
        if submit_question:
            with st.spinner("processing..."):
                answer = process_question(st.session_state['vector_store'],question)
                st.markdown(f"**Question:** ")
                st.write(answer['query'])
                st.markdown(f"**Response:** ")
                st.write(answer['result'])
                st.session_state.vid_chat_history.append((question, answer['result']))
        
        with st.sidebar:    
            clear_chat_history = vid_clear_button()
            if clear_chat_history:
                clear_vid_chat_history()
                reset_session_state()
                st.rerun()
                
            
            st.subheader(f"**Conversation History**")
            for idx, (question, answer) in enumerate(reversed(st.session_state.vid_chat_history)):
                with st.expander(f"Q: {question}"):
                    st.markdown("**Question:**")
                    st.write(question)
                    st.markdown("**Answer:**")
                    st.write(answer)        
            

if __name__== '__main__':
    main()

