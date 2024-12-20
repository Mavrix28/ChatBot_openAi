import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'Q&A Chatbot'

# Streamlit UI
st.title('Q&A Chatbot')

# Sidebar for settings
with st.sidebar:
    api_key = st.text_input('OpenAI API Key', type='password')
    selected_model = st.selectbox('LLM', ['gpt-4', 'gpt-3.5-turbo'])
    temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider('Max Tokens', min_value=50, max_value=300, value=150, step=1)

# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful assistant. Please respond to user queries."),
        HumanMessage(content="{question}")
    ]
)

# Function to generate response
def generate_response(question, api_key, llm_name, temperature, max_tokens):
    # Ensure OpenAI API key is set
    if not api_key:
        return "API key is missing!"

    openai.api_key = api_key

    try:
        # Initialize ChatOpenAI with parameters
        llm = ChatOpenAI(
            model=llm_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key
        )

        # Chain setup
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run(question=question)
        return response

    

    except API_Error as e:
        return f"API Error: {str(e)}"

    except InvalidRequestError as e:
        return f"Invalid Request Error: {str(e)}"

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# User input
st.write('Go ahead, ask me anything!')
user_input = st.text_input('Your question')

if user_input:
    if not api_key:
        st.error('Please provide an OpenAI API Key in the sidebar.')
    else:
        with st.spinner('Generating response...'):
            try:
                answer = generate_response(user_input, api_key, selected_model, temperature, max_tokens)
                if "Invalid API key" in answer:
                    st.error(answer)
                elif "OpenAI Error" in answer:
                    st.warning(answer)
                else:
                    st.success('Response generated successfully!')
                    st.write(answer)
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")