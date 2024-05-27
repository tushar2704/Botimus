##© 2024 Tushar Aggarwal. All rights reserved.(https://tushar-aggarwal.com)
##Botimus[Towards-GenAI] (https://github.com/Towards-GenAI)
##################################################################################################
#Importing dependencies
# Importing dependencies
import streamlit as st
from pathlib import Path
import base64
import sys
import os
import logging
import warnings
from dotenv import load_dotenv
from typing import Any, Dict
import google.generativeai as genai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Crew, Process, Agent, Task
from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
#Importing from SRC
# from src.crews.agents import researcher, insight_researcher, writer, formater
# from src.crews.task import research_task, insights_task, writer_task, format_task
# from src.crews.crew import botimus_crew
from src.components.navigation import footer, custom_style, page_config
##################################################################################################
##################################################################################################
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
##################################################################################################
##################################################################################################
#Environmental variables For developing and testing
# load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")
##################################################################################################
##################################################################################################

#Check if api key loaded successfully with logging info For developing and testing
# if google_api_key:
#     logger.info("Google API Key loaded successfully.")
# else:
#     logger.error("Failed to load Google API Key.")
##################################################################################################
#Intializing llm
page_config("Botimus", "🤖", "wide")
custom_style()
st.sidebar.image('./src/logo.png')
google_api_key = st.sidebar.text_input("Enter your GeminiPro API key:", type="password")

llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, 
                             temperature=0.2, google_api_key=google_api_key)


##################################################################################################


# Custom Handler for logging interactions
class CustomHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name

    def on_chain_start(self, serialized: Dict[str, Any], outputs: Dict[str, Any], **kwargs: Any) -> None:
        st.session_state.messages.append({"role": "assistant", "content": outputs['input']})
        st.chat_message("assistant").write(outputs['input'])

    def on_agent_action(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
        st.chat_message("assistant").write(inputs['input'])

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
        st.chat_message(self.agent_name).write(outputs['output'])

# Main function to run the Streamlit app
def main():
    st.title("🤖Botimus🤖")
    st.markdown('''
            <style>
                div.block-container{padding-top:0px;}
                font-family: 'Roboto', sans-serif; /* Add Roboto font */
                color: blue; /* Make the text blue */
            </style>
                ''',
            unsafe_allow_html=True)
    st.markdown(
        """
        ### Write Structured blogs with AI Agents, powered by Gemini Pro & CrewAI  [Towards-GenAI](https://github.com/Towards-GenAI)
        """
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Input for blog topic
    blog_topic = st.text_input("Enter the blog topic:")

    # Dropdown for selecting the type of content
    content_type = st.selectbox("Select the type of content:", ["Blog Post", "Research Paper", "Technical Report"])

    # Create agents
    researcher = Agent(
        role='Senior Researcher',
        goal='Discover groundbreaking technologies',
        backstory='A curious mind fascinated by cutting-edge innovation and the potential to change the world, you know everything about tech.',
        verbose=True,
        tools=[search_tool],
        allow_delegation=False,
        llm=llm
    )

    insight_researcher = Agent(
        role='Insight Researcher',
        goal='Discover Key Insights',
        backstory='You are able to find key insights from the data you are given.',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    writer = Agent(
        role='Tech Content Strategist',
        goal='Craft compelling content on tech advancements',
        backstory='You are a content strategist known for making complex tech topics interesting and easy to understand.',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    formater = Agent(
        role='Markdown Formater',
        goal='Format the text in markdown',
        backstory='You are able to convert the text into markdown format',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Define tasks with expected_output
    research_task = Task(
        description=f'Identify the next big trend in AI related to {blog_topic} by searching the internet',
        agent=researcher,
        expected_output='A list of potential AI trends'
    )

    insights_task = Task(
        description=f'Identify few key insights from the data related to {blog_topic} in points format. Don’t use any tool',
        agent=insight_researcher,
        expected_output='Key insights in bullet points'
    )

    writer_task = Task(
        description=f'Write a short {content_type} about {blog_topic} with subheadings. Don’t use any tool',
        agent=writer,
        expected_output=f'A well-structured {content_type.lower()}'
    )

    format_task = Task(
        description='Convert the text into markdown format. Don’t use any tool',
        agent=formater,
        expected_output='Formatted markdown text'
    )

    # Instantiate the crew
    tech_crew = Crew(
        agents=[researcher, insight_researcher, writer, formater],
        tasks=[research_task, insights_task, writer_task, format_task],
        process=Process.sequential  # Tasks will be executed one after the other
    )

    # Button to start the task execution
    if st.button("Start Blogging Crew"):
        if blog_topic:
            result = tech_crew.kickoff()
            st.write(result)
        else:
            st.error("Please enter a blog topic.")
            
    with st.sidebar:
        
        footer()

if __name__ == "__main__":
    main()