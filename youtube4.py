import os
import streamlit as st
from langchain import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import YoutubeLoader
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
import tempfile

# Access open AI key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Styles Setup  #########################################################################

# Define header size/color:

def header():
    colored_header(
        label ="YouTube Chat Assistant",
        description = "Find a YouTube video with accurate captions. Enter url below.",
        color_name='light-blue-40'   
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
                background-color: #74eeff;
                color: #000000;
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
                background-color: #74eeff;
                color: #000000;
                border-radius: 20px;
                }
                """
    ) as container:
        return st.button("Submit question")

def clear_button():
    with stylable_container(
        key="clear",
        css_styles="""
            button {
                background-color: #74eeff;
                color: #000000;
                border-radius: 20px;
                }
                """
    ) as container:
        return st.button("Clear all")
    
# End styles section ###########################################################################

# Define functions #############################################################################

# Define function to clear history
def clear_history():
    st.session_state['history'] = []

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

def handle_question(question, crc, history):
    response = crc.run({'question': question, 'chat_history': history})
    history.append((question, response))
    return response, history

def display_response(response, history):
    st.write(response)
    st.divider()
    st.markdown(f"**Conversation History**")
    for prompts in reversed(history):
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


# def clear_persistent_data(directory):
#     """ Remove all files from the specified persistent data directory """
#     """Ensure no other programs are accessing the files before attempting to delete."""
#     time.sleep(15)  # Add a delay to ensure any locks or active usage on the files are released
    
#     for proc in psutil.process_iter(['pid', 'name']):
#         try:
#             files = proc.open_files()
#             for file in files:
#                 if directory in file.path:
#                     print(f"file in use by process: {proc.info} - {file.path}")
#                     return
#         except Exception as e:
#             print(f"Error: {str(e)}")
            
#     try:
#         shutil.rmtree(directory)
#         os.makedirs(directory)  # Recreate the directory after clearing it
#         print("Persistent data cleared and directory reset.")
#     except Exception as e:
#         print(f"Error clearing persistent data: {str(e)}")
        
def process_new_video(youtube_url):
    """ Handles loading, splitting, and embedding a new video """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        loader = YoutubeLoader.from_youtube_url(youtube_url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=temp_dir)
        st.session_state['vector_store'] = vector_store
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=.2)
        retriever = vector_store.as_retriever()
        crc = ConversationalRetrievalChain.from_llm(llm, retriever)
        st.session_state['crc'] = crc

    return vector_store, crc

# Define main function
def main():
    
    header()
    youtube_url = st.text_input('Input YouTube URL')
    process_video = video_button()
    
    # Initialize vector_store and crc
    vector_store = st.session_state.get('vector_store')
    crc = st.session_state.get('crc')
    
    # initialize history
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    # if 'crc' not in st.session_state:
    #     st.session_state['crc'] = []
        
    # if 'vector_store' not in st.session_state:
    #     st.session_state['vector_store'] = []
        
    if process_video and youtube_url:
        # Clear relevant session state keys for new video processing
        reset_session_state(keys=['history', 'vector_store', 'crc'])
        # clear_persistent_data('db2')
        
        with st.spinner('reading, chunking, and embedding...'):
            vector_store, crc = process_new_video(youtube_url)
            st.success('Video processed and ready for queries.')
        
            
    question = st.text_area('Input your question')
    
    col1, col2 = st.columns(2)
    with col1:
        submit_question = question_button_and_style()
    with col2:
        if clear_button():
            reset_session_state()
            # clear_persistent_data('db2')
            st.experimental_rerun()
    
    if submit_question:
        if 'crc' in st.session_state:
            response, updated_history = handle_question(question, st.session_state['crc'], st.session_state['history'])
            st.session_state['history'] = updated_history
            display_response(response, updated_history)

if __name__== '__main__':
    main()

