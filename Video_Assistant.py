

import os
import streamlit as st
from langchain_openai import OpenAI, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import YoutubeLoader
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container


# Access open AI key
OPENAI_API_KEY = st.secrets["YOUTUBE_OPENAI_API_KEY"]
openai_embed_model = "text-embedding-ada-002"
openai_model = "gpt-3.5-turbo-16k"
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
            font-size: 60px !important; /* Change this value to increase or decrease font size
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        /* Adjust the thickness of the line */
        hr {
            height: 16px !important; /* Change this value to increase or decrease line thickness
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

def process_video(url):
    loader = YoutubeLoader.from_youtube_url(url)
    documents = loader.load()
    if not documents:
        st.error("No captions found in video. Please try a different video.")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        st.error("Failed to generate text chunks from the video captions.")
        return None
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model = "text-embedding-ada-002")
    vector_store = FAISS.from_documents(chunks, embedding)
    return vector_store

def process_question(vector_store, question):
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm = llm, chain_type='stuff', retriever=retriever)
    results = qa.invoke(question)
    return results


# Define main function
def main():
    
    header()       
        
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
    
    question = st.text_input('Input your question')
    submit_question = question_button_and_style()
    
    if submit_question:
        with st.spinner("processing..."):
            answer = process_question(st.session_state['vector_store'],question)
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

