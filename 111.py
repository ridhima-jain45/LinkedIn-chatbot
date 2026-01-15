import llama_index
from llama_index.core import Settings, VectorStoreIndex, Document, SimpleDirectoryReader, load_index_from_storage, PromptTemplate
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.experimental.query_engine import PandasQueryEngine
# Try importing from the new dedicated agents package structure
# NOTE: You must install 'llama-index-agents-react' first if this is the case:
# pip install llama-index-agents-react
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import streamlit as st
import pandas
import json
import os
# from prompt import * # <-- REMOVED: Definitions are now included below!
import chromadb 

# --- PROMPT DEFINITIONS (MOVED FROM prompt.py) ---

instruction_str = """\
1. If the query requests information about a single person (e.g., 'give me details of John Doe'), write a single Python expression using Pandas to answer the query and return a dictionary or JSON-like structure containing the complete information of that person.
2. If the query asks for a list of people based on specific criteria (e.g., 'give a list of people from New York'), write a single Python expression that returns a list of dictionaries, with each dictionary containing the complete information of each person who matches the criteria.
3. Ensure the code can be executed with eval() by keeping it as a single expression.
4. The expression should ignore case sensitivity when checking any column.
5. Only output the final expression without quotation marks.
6. The columns in the DataFrame are: 'timestamp', 'id', 'name', 'city', 'country_code', 'region', 'current_company:company_id', 'current_company:name', 'position', 'following', 'about', 'posts', 'groups', 'current_company', 'experience', 'url', 'people_also_viewed', 'educations_details', 'education', 'avatar', 'languages', 'certifications', 'recommendations', 'recommendations_count', 'volunteer_experience', 'courses'.
7. In the dataframe, column 'city' refers to the country
8. In the dataframe, column 'current_company:name' refers to the company.
9. In the dataframe, column 'about' refers to the about the person.
10. In the dataframe, column 'experience' refers to the experience.
11. In the dataframe, column 'url' refers to the linkeldn profile.
12. In the dataframe, column 'education' refers to the qualifications.
13. In the dataframe, column 'educations_details' refers to the education information.
14. In the dataframe, column 'languages' refers to the language proficiency or languages known.
15. In the dataframe, column 'certifications' refers to the skill or certifications.
16. In the dataframe, column 'recommendations' refers to the recommendation.
""" 

new_prompt = PromptTemplate( 
    """\
    You are working with a pandas DataFrame in Python. 
    The name of the DataFrame is df. 
    This is the result of print(df.head()): 
    {df_str} 

    Follow these instructions: 
    {instruction_str} 

    Query: 
    {query_str} 

    Expression: 
    """ 
) 

context = """ 
Purpose: The primary role of this agent is to assist users by providing accurate and 
comprehensive information about LinkedIn profiles of people. 
The agent should be able to handle complex queries related to any of the fields in the 'population.csv' file 
and return the complete information of the person or a list of people as dictionaries or JSON-like structures 
based on the query criteria.
"""
# -------------------------------------------------------------------

# --- Function to safely load API Key ---
def get_api_key(file_name):
    """Reads the API key from a specified file."""
    try:
        with open(file_name, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        st.error(f"Error: API key file '{file_name}' not found. Please create it.")
        return None

# --- LLM and Settings Configuration ---
api_key = get_api_key("key.txt")
if not api_key:
    # If API key is missing, stop script execution here
    st.stop()
    
# FIX 1: Use GoogleGenAI (the imported class) instead of the old 'Gemini'
llm = GoogleGenAI(
    api_key=api_key,
    model_name="gemini-2.5-flash"  
)
Settings.llm = llm


# --- Data Loading and Query Engine Setup ---
profile_path = r"C:\Users\ridhi\Documents\Documents\AI PROJECT\LinkedIn people profiles datasets.csv"
try:
    profile_df = pandas.read_csv(profile_path)
except FileNotFoundError:
    st.error(f"Error: Data file not found at path: {profile_path}")
    st.stop()
    

profile_query_engine = PandasQueryEngine(
    llm=llm, 
    df=profile_df, 
    verbose=True,
    instruction_str=instruction_str # Now accessible
)
profile_query_engine.update_prompts({"pandas_prompt": new_prompt}) # Now accessible
    
# --- Agent Tool Definitions ---
tools = [
    QueryEngineTool(
        query_engine=profile_query_engine,
        metadata=ToolMetadata(
            name="linkedin_profiles_data",
            description=(
                "This tool provides information about LinkedIn profiles based on their city and other attributes. "
                "Use this tool for any query involving the LinkedIn people profiles dataset."
            ),
        ),
    ),
    # ... (other tools)
]

# --- Initialize ReAct Agent ---
# The ReActAgent.from_tools method is now called correctly after resolving the NameError
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context) # 'context' is now accessible


# --- Streamlit UI Components ---
st.markdown(""" 
    <style> 
    /* Streamlit UI customizations */
    [data-testid="stAppViewContainer"]{
        background-image: linear-gradient(to left, #000000, #000000 , #00008b); 
    } 
    .title { 
        color: #4A90E2; 
        text-align: center; 
        margin: 20px 0; 
        font-size: 4em; 
        position: relative;
        top: 10%;
    } 
    .description { 
        font-size: 1.2em; 
        text-align: center; 
        margin-bottom: 20px; 
        color: #E0E0E0; 
    } 
    .stTextInput input:focus { 
        border: 2px solid #4A90E2 !important;
        border-radius: 15px;      
        box-shadow: 0 0 5px #4A90E2 !important; 
        outline: none;     
    } 
    .output {    
        border-radius: 5px;
        background-color: #333333; 
        color: #FFFFFF;
        padding: 15px; 
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5); 
        margin-top: 10px; 
    } 
    .question { 
        font-weight: bold; 
        margin-bottom: 10px; 
        color: #FFFFFF;
    } 
    .response { 
        font-weight: bold; 
        margin-bottom: 10px; 
        color: #4A90E2;
    }
    .image{
        width: 900px;
        height: 100px;    
        margin: 0px;
    }
    </style> 
""", unsafe_allow_html=True) 

# Display Header and Title
st.markdown('<img src="https://www.logo.wine/a/logo/LinkedIn/LinkedIn-Logo.wine.svg" class="image"/>', unsafe_allow_html=True)
st.markdown('<div class="title">Job Profile Bot</div>', unsafe_allow_html=True) 
st.markdown('<div class="description">Get information from LinkedIn profiles based on specific queries!</div>', unsafe_allow_html=True) 
st.sidebar.header('Chat history:')

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# User input for queries 
prompt = st.text_area("Get employee information:", "", key="input", placeholder="Enter your requirements") 

# Helper function to display structured DataFrame results
def display_profile_info(profile_info): 
    # This helper is kept simple for now; agent output will typically be a string summary.
    if isinstance(profile_info, (dict, list)):
        st.json(profile_info)
    else:
        st.write(profile_info)

# --- Query Logic and History Management ---
if prompt:  
    try:  
        # Display user question
        st.markdown(f'<div class="question">Your Question:</div> <div class="output">{prompt}</div>', unsafe_allow_html=True)  
 
        # Run the agent query
        with st.spinner('Thinking...'):
            result = agent.query(prompt)  
        
        # Store the result in session history 
        st.session_state.history.append({"prompt": prompt, "result": result}) 
        
        # Display response
        st.write('') 
        st.markdown('<div class="response">Response:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="output">{result}</div>', unsafe_allow_html=True)

    except Exception as e:  
        st.error(f'<div class="error">An error occurred while processing your query: {str(e)}</div>', unsafe_allow_html=True)  

# --- Sidebar History Display ---
prompt_options = ["Select a question..."] + [entry['prompt'] for entry in st.session_state.history] 

selected_prompt = st.sidebar.selectbox("Select a question to view the answer:", prompt_options) 

if selected_prompt != "Select a question...": 
    for entry in st.session_state.history: 
        if entry['prompt'] == selected_prompt: 
            st.sidebar.markdown(f"**Question:** {entry['prompt']}") 
            
            # Display the associated result 
            st.sidebar.markdown("**Answer:**")
            
            # Check the type of the result before displaying
            if isinstance(entry["result"], str): 
                st.sidebar.markdown(entry['result']) 
            elif isinstance(entry["result"], (list, dict)): 
                st.sidebar.json(entry["result"]) # JSON display for structured data 
            else: 
                st.sidebar.markdown(str(entry['result'])) 
            break