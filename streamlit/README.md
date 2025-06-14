# **Accounting-AI proof-of-concept apps:**
The accounting field is an excellent candidate for AI support; it is rife with repetitive, labor-intensive tasks that are performed monthly/quarterly/annually.  But where to start? And how do we rely on something so unknown for such important and sensitive things as financials?

While the AI landscape is changing so rapidly, it's hard to keep up, and it's hard to know which tools are most effective, or trustworthy. Investing in a major AI tool now, that requires planning and implementation, could result in a tool that is already obsolete or replaced by more effective tools by the time you get it live in your environment. Not to mention security considerations and policies that need to be established.  Companies are struggling to keep up.  But that doesn't mean you can't start dipping your toes in the water and building some AI muscles. 

## **Start small, find low-risk pain points, and start tinkering** 
This also starts building the infrastructure and experience for teams to think creatively about how to use AI. I created this collection of apps to showcase my skills, and also to help accountants visualize some ways in which small and accessible tools can help save time and effort on your teams.  

The tools I've shared here can be hosted on internal systems, the vector databases can be stored on internal IT infrastructure, and the apps can call internal LLM instances (Llama 3 is now available for download for free), or can access Open AI through enterprise level APIs with enhanced privacy and security commitments.  
 
## **Lease Assistant:** 
Chat assistant trained on lease accounting guidance under ASC 842.

   **Potential use cases:**
   - Users ask technical accounting questions and receive responses grounded in the guidance with references of where to look for more information.
   - Would it replace the technical accounting team? No.
   - Should it be relied upon to make major accounting decisions? No.
   - Not uncommon for accounting team members to need information about a particular accounting topic, but the technical accounting team doesn't have capacity to support. It can require substantial time and effort to find the correct guidance, and find the right topic within the guidance. But, a chat assistant like this could give general information, and provide references, saving people time and effort.
   - This concept could be specialized for many different scenarios: internal company policies, internal accounting memos and procedures, user guides for new implementations and software. It can allow employees to self-serve for basic questions without having to navigate complicated SharePoint repositories.

  **Example queries:**
   - What are lease components and nonlease components, and how are they identified?
   - What are the journal entries required to account for a lease for lessees and lessors?
   - What are the steps required in considering a lease modification?
   - I don't understand what a sales-type lease is, can you help me understand and explain it like I'm a child?

**Features:**

 - AI driven conversational interface
 - Trained specifically on technical accounting guidance and instructed to only answer relevant lease accounting relevant questions
 - Internal prompt includes example responses and instructions for response and reference formatting
 - Uses gpt-3.5-turbo for generating responses
 - Embeddings are stored in vector store, enabling rapid searches

**Technologies used:**
 - **Streamlit**: To create the web interface and community cloud hosting
 - **LangChain**: Foundational framework connecting OpenAI's models and Qdrant vector storage
   - **create_history_aware_retriever**
   - **create_stuff_documents_chain**
   - **create_retrieval_chain**
 - **OpenAI's gpt-3.5-turbo**: For natural language understanding and response generation
 - **OpenAI's text-embedding-3-large**: To create source content embeddings
 - **Qdrant**: For efficient vector-based document retreival

**Known issues:**
 - Lease assistant does not handle follow-up questions well. User must currently enter a full question with all necessary context each time question is submitted. This is solvable to allow for follow up questions.
 - I initially tried to include page references to the guidance because the page number metadata is included in the embeddings sent to the vector store. However, the page refereces provided were consistently inaccurate, so I removed that feature. 
 - Some responses do not provide complete information or include hallucinations.  Currently prompt template includes only two example training responses. Working on adding additional training materials to increase accuracy/completeness.

## **YouTube Chat Assistant: (note: I have disabled the YouTube Chat Assistant as YouTube now blocks the API call it leverages, but I have left the documentation here to showcase the thought behind building this app and why it would be relevant in an accounting or corporate environment)** 
Ingests YouTube videos (with captions). Users can ask both broad and specific questions about the video.

   **Potential use cases:**
   - Enter training session video and ask for step-by-step instructions on how to do the processes in the video. Could be used to create SOPs from onboarding training videos.
   - Ask the chatbot to provide a bulleted list of the main talking points with a summary of each - for financial webcasts, or a meeting, that someone might not have time to watch in full, but wants to know the general summary.
   - Walkthroughs with auditors - record the session, then feed it through a chatbot assistant to document steps, or ask follow-up questions if details are forgotten.
   - Train the chatbot on a full repository of onboarding training videos for new hires.  When users ask questions, it provides reponses as well as link to specific video and links to other internal documents with further information.

  **Example queries:**
   - Give me a bulleted list of the main talking points with a summary of each.
   - Provide step-by-step instructions for the process described in the video.
   - Where in the video did the content about [x] get discusses, the beginning, a quarter of the way through, half way through, 3 quarters of the way through, or at the end? (note, it can't give timestamp data, but can give a general sense of where in the video specific content is)
   - What is the most important point of the video?
   - For a video of a legislative session: give me a bulleted list of all the bills voted on during the session. Include bill numbers and bill names, what the vote outcome was, and a summary of each bill.


**Features:**

 - AI driven conversational interface
 - Uses gpt-3.5-turbo-16k for generating responses
 - Embeddings are stored in vector store, enabling rapid searches

**Technologies used:**
 - **Streamlit**: To create the web interface and community cloud hosting
 - **LangChain**: Foundational framework connecting OpenAI's models and FAISS vector storage
 - **OpenAI's gpt-3.5-turbo-16k**: For natural language understanding and response generation
 - **OpenAI's text-embedding-ada-002**: To create source content embeddings
 - **FAISS**: To store vectors and index them for fast retrieval

## Getting Started

### Prerequisites
**Python:** Version 3.11 or newer is required. This application has been tested on Python 3.11.5. Make sure your Python version is up-to-date to avoid compatibility issues.

### Installation
To set up the project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mmccanse/accounting-ai.git

2. **Navigate to the repository folder:**
   ```bash
   cd accounting-ai

3. **Install the required dependencies::**
   ```bash
   pip install -U langchain-community
   pip install faiss-cpu
   pip install langchain
   pip install langchain-community
   pip install langchain-openai
   pip install openai
   pip install python-dotenv
   pip install qdrant_client
   pip install tiktoken
   pip install streamlit
   pip install streamlit_extras
   pip install youtube-transcript-api

## Usage

To run the application:

1. Open a terminal in the project directory.
2. Start the program by executing:
   ```bash
   streamlit run streamlit/Lease_Assistant.py
 -or-
1. Open repository in VS code
2. Right-click on Video_Assistant.py
3. Open in integrated terminal and execute:
   ```bash
   streamlit run streamlit/Lease_Assistant.py

