
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAI
from qdrant_client import QdrantClient, models
import qdrant_client
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from streamlit_extras.stylable_container import stylable_container
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

tab1, tab2 = st.tabs(["Lease Assistant", "Video Assistant"])

with tab1:

    # Set the model name for LLM
    OPENAI_MODEL = "gpt-3.5-turbo"

    # Store API key as a variable
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    # Define function to access Qdrant vector store
    def get_vector_store():
        #Create a client to connect to Qdrant server
        client = qdrant_client.QdrantClient(
            st.secrets["QDRANT_HOST"],
            api_key=st.secrets["QDRANT_API_KEY"]
            )
        
        #initialize embeddings for vector store
        embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model="text-embedding-3-large"
        )
        
        #create a vector store with Qdrant and embeddings
        vector_store = Qdrant(
            client = client,
            collection_name = st.secrets["QDRANT_COLLECTION_NAME"],
            embeddings = embeddings,
        )
        
        return vector_store

    # Define context (not sure if it causes issues that this is separate and in the prompt template)
    context = """You are an expert leases chatbot. You answer questions relating to ASC 842 under US GAAP. Users rely on you to be accurate and thorough in your responses. Please follow the following instructions in constructing your responses:
    1.	You respond to the queries as shown in the provided examples. The responses do not have to be brief. Giving a thorough response is more important than brevity. 
    2.	Each response will be followed by a list of references. The references will be a list of each source document and its relevant page number. The page number will be obtained from the meta data in the vector store.
    3.	If the source material is sourced from a range of pages, include a page range as the reference.
    4.	Your response will also include a separate reference to the relevant ASC 842 guidance chapter.  
    5.	If your response refers to an example provided, the response needs to include the full example being referenced.
    6.	Your responses will be provided only from the provided vector store source context documents. 
    7.	Your responses will be clear and helpful and will use language that is easy to understand. 
    8.	Your responses will include examples and potential scenarios.  
    9.	If the answer is not available in the vector store source documents, the response will be "I can share general knowledge about lease accounting, but I cannot advise on specific scenarios, please seek guidance from a qualified expert." 
    10.	If the question is not on the topic of leases, respond by saying, "This is outside the scope of what I can help you with. Let's get back to lease accounting."""


    # Create function to setup prompt template
    def setup_prompt_template():
        prefix="""You are an expert leases chatbot. You answer questions relating to ASC 842 under US GAAP. Users rely on you to be accurate and thorough in your responses. Please follow the following instructions in constructing your responses:
    11.	You respond to the queries as shown in the provided examples. The responses do not have to be brief. Giving a thorough response is more important than brevity. 
    12.	Each response will be followed by a list of references. The references will be a list of each source document and its relevant page number. The page number will be obtained from the meta data in the vector store.
    13.	If the source material is sourced from a range of pages, include a page range as the reference.
    14.	Your response will also include a separate reference to the relevant ASC 842 guidance chapter.  
    15.	If your response refers to an example provided, the response needs to include the full example being referenced.
    16.	Your responses will be provided only from the provided vector store source context documents. 
    17.	Your responses will be clear and helpful and will use language that is easy to understand. 
    18.	Your responses will include examples and potential scenarios.  
    19.	If the answer is not available in the vector store source documents, the response will be "I can share general knowledge about lease accounting, but I cannot advise on specific scenarios, please seek guidance from a qualified expert." 
    20.	If the question is not on the topic of leases, respond by saying, "This is outside the scope of what I can help you with. Let's get back to lease accounting.
    
        
    You will answer the input question based on the provided context:
        
        <context>
        {context}
        </context>
        
    You will use the provided examples for guidance on how to construct your responses. Your responses should be similar and the format you use to provide the references should be the exact format for references in the examples.
    """
        
        # Define examples to instruct app how to respond
        examples = [
            {
                "input": "How do I determine the different lease components?",
                "answer": """To identify the different lease and nonlease components in a lease contract, both lessees and lessors need to follow specific steps outlined in ASC 842.
    For lessees, lease components are elements of the arrangement that provide the lessee with the right to use an identified asset. The right to use an underlying asset is considered a separate lease component if the lessee can benefit from the asset either on its own or together with other readily available resources, and if the asset is not highly dependent on or highly interrelated with other assets in the contract. This determination is crucial in allocating consideration between lease and nonlease components.
    For lessors, the process is similar but with some differences. Lessors must allocate any capitalized costs, such as initial direct costs or contract costs, to the separate lease components or nonlease components to which those costs relate. Lessors also have the option to elect a practical expedient where they can choose not to separate nonlease components from lease components and account for them as a single component under certain conditions.
    To provide a comprehensive understanding, let's look at an example from the KPMG Leases Handbook:
    Example: A lessor enters into a lease contract with a lessee for the use of equipment. The contract includes maintenance services as a nonlease component. The lessor must identify the separate lease component (equipment) and nonlease component (maintenance services) and allocate consideration accordingly. If the maintenance services are considered a predominant component, the lessor may account for them under Topic 606. Otherwise, the lessor would account for the combined component as an operating lease under ASC 842.

                References: 
                KPMG Leases Handbook, page 151
                PWC Leases Guide, pages 47 - 72
                EY Financial Reporting Developments Lease Accounting, pages 41 - 73
                ASC: 842-10-15-28 to 842-10-15-42"""

            },
            {
                "input": "How do I account for lease modifications?",
                "answer": """When accounting for lease modifications, both lessees and lessors have specific considerations to take into account. Let's break down the accounting treatment for both parties:

    For lessees, when a lease modification occurs, the lessee must assess whether the modification results in a separate lease or not. If it does, the lessee will account for the modification as a separate lease. This involves recognizing a new right-of-use asset and lease liability based on the remeasured consideration for the modified lease.

    If the modification does not result in a separate lease, the lessee will need to remeasure the lease liability and adjust the right-of-use asset based on the remaining consideration in the contract. The lessee must also reassess the lease classification at the modification effective date and account for any initial direct costs, lease incentives, and other payments made to or by the lessee in connection with the modification.

    Overall, the accounting treatment for lease modifications for lessees involves careful consideration of the impact on the financial statements and compliance with ASC 842 requirements.
    References:
    •	EY Financial Reporting Developments Lease Accounting, pages 204-226
    •	PWC Leases Guide, pages 182 - 225 
    •	KPMG Leases Handbook, pages 567 - 609
    •	ASC 842-10-25-8 to 25-14, 842-10-35-3 to 35-6

    For lessors, accounting for lease modifications involves several steps and considerations. When a lease modification occurs, the lessor must first determine if the modified contract is still considered a lease or contains a lease. If it does, the lessor needs to assess whether the modification results in a separate contract or not.
    If the modification results in a separate contract, the lessor will account for two separate contracts: the unmodified original contract and the new separate contract. The new separate contract is accounted for in the same manner as any other new lease. This means that the lessor will recognize any selling profit or loss on the modified lease based on the fair value of the underlying asset at the effective date of the modification.
    On the other hand, if the modification does not result in a separate contract, the lessor will need to remeasure and reallocate the remaining consideration in the contract. The lessor must also reassess the lease classification at the modification effective date and account for any initial direct costs, lease incentives, and other payments made to or by the lessor in connection with the modification.
    The accounting treatment for lease modifications will vary depending on whether the lease classification changes and how it changes. It is essential for lessors to carefully evaluate each modification to ensure compliance with ASC 842 guidelines.
    References:
    •	EY - Financial Reporting Developments: Lease Accounting, pages 281-298
    •	PWC - Leases Guide, pages 226 - 241
    •	KPMG - Leases Handbook, pages 709 - 735
    •	ASC: 842-10-25-8 to 25-18, 842-10-35-3 to 35-6, 842-10-55-190 to 55-209"""

            }
        ]
        
        #Define format for examples:
        example_format = "\nQuestion: {input}\n\nAnswer: {answer}"
        
        example_prompts = [example_format.format(**ex) for ex in examples]
        
        example_template = PromptTemplate(input_variables=['input', 'context'],
                                        template=example_format)
        
        full_prompt = f"{prefix}\n\n" + "\n\n".join(example_prompts) + "\n\nQuestion: {input}\n\nAnswer: "
        
        # enriched_history = history + [(input, full_prompt)]
        
        #Define suffix for query
        suffix="\n\nQuestion: {input}\nAnswer: "
        
        #Construct FewShotPromptTemplate
        prompt_template = FewShotPromptTemplate(
                                                examples=examples,
                                                example_prompt=example_template,
                                                input_variables=['input','context'],
                                                prefix=prefix,
                                                suffix=suffix,
                                                example_separator="\n\n")
        return prompt_template
        
        
    def create_history_aware_chain(prompt_template,vector_store):    
        # Create new llm instance
        llm = ChatOpenAI(api_key=openai_api_key, model=OPENAI_MODEL, temperature=0.0)
        # Set vector_store as retriever
        retriever = vector_store.as_retriever()
        # create history aware retriever that will retrieve relevant 
        # segments from source docs
        history_aware_retriever_chain = create_history_aware_retriever(
            llm, 
            retriever,
            prompt_template)
        return history_aware_retriever_chain
        

    # create document chain with create_stuff_documents_chain. This 
    # tekes the relevant source segments from the history_aware_retriever
    # and "stuffs" them into (something?) that the retrieval chain will reference.

    def create_document_chain(prompt_template):
        # Create new llm instance
        llm = ChatOpenAI(api_key=openai_api_key, model=OPENAI_MODEL, temperature=0.0)
        doc_chain = create_stuff_documents_chain(llm, prompt_template)
        return doc_chain
        

    def create_retrieve_chain(history_aware_chain, document_chain):
        retrieval_chain = create_retrieval_chain(history_aware_chain, document_chain)
        return retrieval_chain
        
    def display_history():
        with st.sidebar:
            st.subheader("Session History")
            for idx, (input, response) in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Q: {input}"):
                    st.markdown("**Question:**")
                    st.write(input)
                    st.markdown("**Answer:**")
                    st.write(response)

    # define function to clear chat history
    def clear_history():
        if 'history' in st.session_state:
            del st.session_state['history']
        # reset input and answer
        st.session_state['input'] = ""
        
    def submit_button():
        with stylable_container(
            key="sub",
            css_styles="""
                button {
                    background-color: #0e1b22;
                    color: #ffffff;
                    border-radius: 20px;
                    }
                    """
        ) as container:
            return st.button("Submit") 

    # dark pink button color: #a81875

    def clear_button():
        with stylable_container(
            key="clr",
            css_styles="""
                button {
                    background-color: #0e1b22;
                    color: #ffffff;
                    border-radius: 20px;
                    }
                    """
        ) as container:
            return st.button("Clear History")   

    # define streamlit app
    def main():
        st.title('Lease Accounting AI Assistant')
        # st.header('This assistant is still a work in progress. The page number references are not quite dialed in. Thank you for your patience.')
        st.divider()
        st.write('This is a retrieval-assisted chatbot for lease accounting under US GAAP.')
        st.markdown(f"**Disclaimer:** This assistant cannot give specific accounting advice. It is a learning tool and proof of concept. It is not intended for commercial use. PLease note that page number references provided in the response are not currently accurate. For accounting advice, please consult an appropriate professional.")

        st.divider()
        
        try:
                
            #Initialize history before it is accessed
            if 'history' not in st.session_state:
                st.session_state['history'] = []
            
            #Initialize vector store
            if 'vector_store' not in st.session_state:
                st.session_state['vector_store'] = get_vector_store()
                # st.success('vector store loaded!')
            
            #Initialize prompt template
            if 'prompt_template' not in st.session_state:
                st.session_state['prompt_template'] = setup_prompt_template()
            
            #bring context into session state
            if 'context' not in st.session_state:
                st.session_state['context'] = context
            
            #establish 'input_value' so able to clear it
            if 'input_value' not in st.session_state:
                st.session_state['input_value'] = ""
            
            
            user_input = st.text_area("""Ask about lease accounting! The app 
                                    remembers your conversation until you click 'Clear History' in the sidebar""", placeholder='Type your question here...')
            submit_button()
            # submit_button = st.button('Submit')
            st.divider()

            if submit_button and user_input:
                with st.spinner("Searching the guidance..."):
                    history_aware_chain = create_history_aware_chain(st.session_state['prompt_template'],st.session_state['vector_store'])
                    documents_chain = create_document_chain((st.session_state['prompt_template']))
                    retrieval_chain_instance = create_retrieve_chain(history_aware_chain, documents_chain)
                    response = retrieval_chain_instance.invoke({
                        'input': user_input}) 
                        # 'context': st.session_state['context'], 
                        # 'chat_history': st.session_state['history']})
                    modified_response = response['answer'].replace("$", "\$")
                    st.markdown(f"**Question:** ")
                    st.write(response['input'])
                    st.markdown(f"**Response:** ")
                    st.write(modified_response)
                    st.session_state.history.append((user_input, response['answer']))
                    
                
            with st.sidebar:
                clear_chat_history = clear_button()
                if clear_chat_history:
                    st.session_state['history'] = []
                    st.session_state['input_value'] = ""
                    
                st.subheader(f"**Conversation History**")
                for idx, (question, answer) in enumerate(reversed(st.session_state.history)):
                    with st.expander(f"Q: {question}"):
                        st.markdown("**Question:**")
                        st.write(question)
                        st.markdown("**Answer:**")
                        st.write(answer)

            
        except Exception as e:
            #add debugging statement
            st.error(f"An error occurred: {str(e)}")

    if __name__ == "__main__":
        main()



###############################################################################

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
import tempfile
import shutil
import time

with tab2:
    # Access open AI key
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai_embed_model = "text-embedding-3-large"
    openai_model = "gpt-3.5-turbo-16k"
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=openai_model, temperature=0.1)

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

    def process_video(url):
        loader = YoutubeLoader.from_youtube_url(url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
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
        if 'history' not in st.session_state:
            st.session_state['history'] = []
            
        if submit_video and youtube_url:
            vector_store = process_video(youtube_url)
            st.session_state['vector_store'] = vector_store
            st.success("Video processed and vector store created.")
            
        question = st.text_area('Input your question')
        
        col1, col2 = st.columns(2)
        with col1:
            submit_question = question_button_and_style()
        with col2:
            if clear_button():
                youtube_url = st.write("")
                question = st.write("")
                reset_session_state()
                st.rerun()
        
        if submit_question:
            answer = process_question(st.session_state['vector_store'],question)
            st.write(answer['result'])
            st.session_state.history.append((question, answer['result']))
            
                
                

    if __name__== '__main__':
        main()

