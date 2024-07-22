
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
import logging
logging.basicConfig(level=logging.DEBUG)

# Set up Streamlit page configuration
st.set_page_config(page_title=None,
                   page_icon=":dart:",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

# Set the model name for LLM
OPENAI_MODEL = "gpt-4o-mini"

# Store API key as a variable
openai_api_key = st.secrets["LEASE_OPENAI_API_KEY"]

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
2.	If your response refers to an example provided, the response needs to include the full example being referenced.
3.	Your responses will be provided only from the provided vector store source context documents. 
4.	Your responses will be clear and helpful and will use language that is easy to understand. 
5.	Your responses will include examples and potential scenarios.  
6.	If the answer is not available in the vector store source documents, the response will be "I can share general knowledge about lease accounting, but I cannot advise on specific scenarios, please seek guidance from a qualified expert." 
7.	If the question is not on the topic of leases, respond by saying, "This is outside the scope of what I can help you with. Let's get back to lease accounting."""


# Create function to setup prompt template
def setup_prompt_template():
    prefix="""You are an expert leases chatbot. You answer questions relating to ASC 842 under US GAAP. Users rely on you to be accurate and thorough in your responses. Please follow the following instructions in constructing your responses:
1.	You respond to the queries as shown in the provided examples. The responses do not have to be brief. Giving a thorough response is more important than brevity. 
2.	If your response refers to an example provided, the response needs to include the full example being referenced.
3.	Your responses will be provided only from the provided vector store source context documents. 
4.	Your responses will be clear and helpful and will use language that is easy to understand. 
5.	Your responses will include examples and potential scenarios.  
6.	If the answer is not available in the vector store source documents, the response will be "I can share general knowledge about lease accounting, but I cannot advise on specific scenarios, please seek guidance from a qualified expert." 
7.	If the question is not on the topic of leases, respond by saying, "This is outside the scope of what I can help you with. Let's get back to lease accounting.
8.  Your responses will not include page number references to the KPMG guidance or to the ASC guidance. Your page number references to date have been hallucinations and are not reliable.
 
    
You will answer the input question based on the provided context:
    
    <context>
    {context}
    </context>
    
You will use the provided examples for guidance on how to construct your responses. Your responses should be similar.
"""
     
     # Define examples to instruct app how to respond
    examples = [
        {
            "input": "How do I determine the different lease components?",
            "answer": """To identify the different lease and nonlease components in a lease contract, both lessees and lessors need to follow specific steps outlined in ASC 842.
For lessees, lease components are elements of the arrangement that provide the lessee with the right to use an identified asset. The right to use an underlying asset is considered a separate lease component if the lessee can benefit from the asset either on its own or together with other readily available resources, and if the asset is not highly dependent on or highly interrelated with other assets in the contract. This determination is crucial in allocating consideration between lease and nonlease components.
For lessors, the process is similar but with some differences. Lessors must allocate any capitalized costs, such as initial direct costs or contract costs, to the separate lease components or nonlease components to which those costs relate. Lessors also have the option to elect a practical expedient where they can choose not to separate nonlease components from lease components and account for them as a single component under certain conditions.
To provide a comprehensive understanding, let's look at an example from the KPMG Leases Handbook:
Example: A lessor enters into a lease contract with a lessee for the use of equipment. The contract includes maintenance services as a nonlease component. The lessor must identify the separate lease component (equipment) and nonlease component (maintenance services) and allocate consideration accordingly. If the maintenance services are considered a predominant component, the lessor may account for them under Topic 606. Otherwise, the lessor would account for the combined component as an operating lease under ASC 842."""

        },
        {
            "input": "How do I account for lease modifications?",
            "answer": """When accounting for lease modifications, both lessees and lessors have specific considerations to take into account. Let's break down the accounting treatment for both parties:

For lessees, when a lease modification occurs, the lessee must assess whether the modification results in a separate lease or not. If it does, the lessee will account for the modification as a separate lease. This involves recognizing a new right-of-use asset and lease liability based on the remeasured consideration for the modified lease.

If the modification does not result in a separate lease, the lessee will need to remeasure the lease liability and adjust the right-of-use asset based on the remaining consideration in the contract. The lessee must also reassess the lease classification at the modification effective date and account for any initial direct costs, lease incentives, and other payments made to or by the lessee in connection with the modification.

Overall, the accounting treatment for lease modifications for lessees involves careful consideration of the impact on the financial statements and compliance with ASC 842 requirements.


For lessors, accounting for lease modifications involves several steps and considerations. When a lease modification occurs, the lessor must first determine if the modified contract is still considered a lease or contains a lease. If it does, the lessor needs to assess whether the modification results in a separate contract or not.
If the modification results in a separate contract, the lessor will account for two separate contracts: the unmodified original contract and the new separate contract. The new separate contract is accounted for in the same manner as any other new lease. This means that the lessor will recognize any selling profit or loss on the modified lease based on the fair value of the underlying asset at the effective date of the modification.
On the other hand, if the modification does not result in a separate contract, the lessor will need to remeasure and reallocate the remaining consideration in the contract. The lessor must also reassess the lease classification at the modification effective date and account for any initial direct costs, lease incentives, and other payments made to or by the lessor in connection with the modification.
The accounting treatment for lease modifications will vary depending on whether the lease classification changes and how it changes. It is essential for lessors to carefully evaluate each modification to ensure compliance with ASC 842 guidelines."""

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
    
    tab1, tab2 = st.tabs(["The App", "Accounting Use Cases"])
    
    with tab2:
        st.subheader("proof of concept")
        st.write("""It's not uncommon for accounting team members to need information about a particular 
                 accounting topic, but the technical accounting team doesn't have capacity to support. 
                 It can require substantial time and effort to find the correct guidance, and find the 
                 right topic within the guidance. Additionally, staff and senior accountants
                 rarely have extra time amid the month-end close cycle to engage in substantial research projects.
                 This assistant is trained on the accounting guidance documents seen in the sidebar, and instructed with 
                 a Few Shot Prompt Template giving narrow context and example responses.""")
        
        st.markdown("""
            <style>
            /* CSS selector for Streamlit expander headers */
            .st-emotion-cache-1h9usn1 { 
                background-color: #0068ca; /* Light Salmon color; change as needed */
                color: white; /* Text color; change as needed */
            }
            </style>
            """, unsafe_allow_html=True)
        
        with st.expander("To see Few Shot Prompt Template, click here"):
            st.code("""def setup_prompt_template():
    prefix="You are an expert leases chatbot. You answer questions relating to ASC 842 
    under US GAAP. Users rely on you to be accurate and thorough in your responses. 
    Please follow the following instructions in constructing your responses:
    1.You respond to the queries as shown in the provided examples. The responses do not 
    have to be brief. Giving a thorough response is more important than brevity. 
    
    2.If your response refers to an example provided, the response needs to include 
    the full example being referenced.
    
    3.Your responses will be provided only from the provided vector store source 
    context documents. 
    
    4.Your responses will be clear and helpful and will use language that is easy to 
    understand. 
    
    5.Your responses will include examples and potential scenarios.  
    
    6.If the answer is not available in the vector store source documents, the response will 
    be "I can share general knowledge about lease accounting, but I cannot advise on 
    specific scenarios, please seek guidance from a qualified expert." 
    
    7.If the question is not on the topic of leases, respond by saying, "This is outside 
    the scope of what I can help you with. Let's get back to lease accounting."
    
        
    You will answer the input question based on the provided context:
        
        <context>
        {context}
        </context>
        
    You will use the provided examples for guidance on how to construct your responses. 
        
    # Define examples to instruct app how to respond
    examples = [
        {
    "input": "How do I determine the different lease components?",
    "answer": "To identify the different lease and nonlease components in 
    a lease contract, both lessees and lessors need to follow specific steps 
    outlined in ASC 842.
    
    For lessees, lease components are elements of the arrangement that provide the lessee 
    with the right to use an identified asset. The right to use an underlying asset is 
    considered a separate lease component if the lessee can benefit from the asset either 
    on its own or together with other readily available resources, and if the asset is not 
    highly dependent on or highly interrelated with other assets in the contract. This 
    determination is crucial in allocating consideration between lease and nonlease 
    components. For lessors, the process is similar but with some differences. Lessors 
    must allocate any capitalized costs, such as initial direct costs or contract costs, 
    to the separate lease components or nonlease components to which those costs relate. 
    Lessors also have the option to elect a practical expedient where they can choose 
    not to separate nonlease components from lease components and account for them as a 
    single component under certain conditions.
    
    To provide a comprehensive understanding, let's look at an example from the KPMG 
    Leases Handbook:
    
    Example: A lessor enters into a lease contract with a lessee for the use of equipment. 
    The contract includes maintenance services as a nonlease component. The lessor must 
    identify the separate lease component (equipment) and nonlease component 
    (maintenance services) and allocate consideration accordingly. If the maintenance 
    services are considered a predominant component, the lessor may account for them 
    under Topic 606. Otherwise, the lessor would account for the combined component as 
    an operating lease under ASC 842.

            },
            {
    "input": "How do I account for lease modifications?",
    "answer": "When accounting for lease modifications, both lessees and lessors have specific 
    considerations to take into account. Let's break down the accounting treatment for both parties:

    For lessees, when a lease modification occurs, the lessee must assess whether the modification results 
    in a separate lease or not. If it does, the lessee will account for the modification as a separate 
    lease. This involves recognizing a new right-of-use asset and lease liability based on the remeasured 
    consideration for the modified lease.

    If the modification does not result in a separate lease, the lessee will need to remeasure the lease 
    liability and adjust the right-of-use asset based on the remaining consideration in the contract. The 
    lessee must also reassess the lease classification at the modification effective date and account for 
    any initial direct costs, lease incentives, and other payments made to or by the lessee in connection 
    with the modification.

    Overall, the accounting treatment for lease modifications for lessees involves careful consideration 
    of the impact on the financial statements and compliance with ASC 842 requirements.

    For lessors, accounting for lease modifications involves several steps and considerations. When a lease 
    modification occurs, the lessor must first determine if the modified contract is still considered a 
    lease or contains a lease. If it does, the lessor needs to assess whether the modification results in a 
    separate contract or not.
    
    If the modification results in a separate contract, the lessor will account for two separate contracts: 
    the unmodified original contract and the new separate contract. The new separate contract is accounted 
    for in the same manner as any other new lease. This means that the lessor will recognize any selling 
    profit or loss on the modified lease based on the fair value of the underlying asset at the effective 
    date of the 
    modification.
    
    On the other hand, if the modification does not result in a separate contract, the lessor will need to 
    remeasure and reallocate the remaining consideration in the contract. The lessor must also reassess the 
    lease classification at the modification effective date and account for any initial direct costs, lease 
    incentives, and other payments made to or by the lessor in connection with the modification.
    The accounting treatment for lease modifications will vary depending on whether the lease classification 
    changes and how it changes. It is essential for lessors to carefully evaluate each modification to ensure 
    compliance with ASC 842 guidelines.
        }
    ]""")


        st.write("""\t- Would it replace the technical accounting team? No.
                 \n-\tShould it be relied upon to make major accounting decisions? No.
                 \n BUT, it would support learning, give general information, and provide references, saving people time and effort.
                 \n ADDITIONALLY, this concept can be used with any number of topics.""")
        st.markdown("**Examples:**")
        st.write(""" 
   \n- \tUsers ask technical accounting questions and receive responses grounded in guidance with references pointing to resources for more information. Could include any range of topics.
   - Empower employees to self-serve questions on any topic without having to navigate cumbersome SharePoint repositories
     - Internal company policies
     - Internal accounting memos and procedures
     - User guides for new implementations and software. 

  **Example queries:**
   - What are lease components and nonlease components, and how are they identified?
   - What are the journal entries required to account for a lease for lessees and lessors?
   - What are the steps required in considering a lease modification?
   - I don't understand what a sales-type lease is, can you help me understand and explain it like I'm a child?
    
  **Features:**

  - AI driven conversational interface
  - Trained specifically on technical accounting guidance and instructed to only answer relevant lease accounting relevant questions
  - Internal prompt includes example responses and instructions for response and reference formatting
  - Uses gpt-4o-mini for generating responses
  - Embeddings are stored in vector store, enabling rapid searches

**Technologies used:**
 - **Streamlit**: To create the web interface and community cloud hosting
 - **LangChain**: Foundational framework connecting OpenAI's models and Qdrant vector storage
   - **create_history_aware_retriever**
   - **create_stuff_documents_chain**
   - **create_retrieval_chain**
 - **OpenAI's gpt-4o-mini**: For natural language understanding and response generation
 - **OpenAI's text-embedding-3-large**: To create source content embeddings
 - **Qdrant**: For efficient vector-based document retreival

**Known issues:**
 - Lease assistant does not currently handle follow-up questions well. User must currently enter a full question with all necessary context each time question is submitted. Working on handling follow-up questions.
 - Page references provided are not currently accurate. The source doc page numbers are in the meta data in the vector store. Working on building code to feed in accurate page references.
 - Some responses do not provide complete information or include hallucinations.  Currently prompt template includes only two example training responses. Working on adding additional training materials to increase accuracy/completeness.""")
    
    
    
    with tab1:
        st.write('This is a retrieval-assisted chatbot for lease accounting under US GAAP.')
        st.markdown(f"**Disclaimer:** This assistant cannot give specific accounting advice. It is a learning tool and proof of concept. It is not intended for commercial use. Please note that any page number references provided in the response are not currently accurate. For accounting advice, please consult an appropriate professional.")

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
