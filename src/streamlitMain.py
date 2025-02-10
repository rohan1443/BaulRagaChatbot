import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from pinecone import ServerlessSpec, Pinecone
from langchain.agents import Tool
from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from RAG_ChatBot import ChatBot
import time
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime

load_dotenv()

# --- Caching and Resource Initialization ---
@st.cache_resource
def init_resources():
    embeddings = OpenAIEmbeddings()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")) 
    index_name = "langchain-demo"
    dimensions = 1536

    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name) 
    pc.create_index(
        name=index_name,
        dimension=dimensions,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    index = pc.Index(index_name)
    return embeddings, index, index_name

embeddings, index, index_name = init_resources()

# --- ChatBot Initialization ---
bot = ChatBot(embeddings, index, index_name)

# --Initialize LLM and Tool for Agent ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
           "system",
            f"""You are a Bengal travel assistant who only speaks about the culture and details of the Bauls of Bengal. 
            Users will ask you questions about their culture, people, folk songs, and instruments. 
            Use the provided context and the tools to fetch the answer for the question. if you dont find any result from the pinecone_tool or if it responds with dont know then use the tavily_tool to fetch the answer from the web.
            if the user asks related images or youtube videos then use the tavily_tool to fetch the answer.
            If you don't know the answer, just say you don't know. 
            Your answer should be short and concise, no longer than 2-3 sentences.
            Do not respond to questions unrelated to the Bauls of Bengal and their culture or folk music. You cab talk about their culture, people, folk songs, instruments, etc.
            But strictly avoid talking about any other topic or ask for calrification if you think the user is asking about any other topic and convey that you are not able to answer the question as it is not related to the Bauls of Bengal.
            **The user's name will be provided at the beginning of their input.**
            Use this name to personalize your greetings in the begining like How can I assist you with providing some exciting info about the rich culture of the Bauls of Bengal
            or ask if you would like to know about the Bauls of Bengal culture, people, folk songs, instruments, etc. and If yes, then give the user some definition about the bauls and ask to ask questions about what you are interested to know about more.
            and to presonalize responses sometimes as well.
            """
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

pinecone_tool = Tool(
    name="pinecone_search",
    func=lambda query: bot.query(query),
    description="Use this tool to search for information about the Bauls of Bengal in the Pinecone vector database."
)

tavily_tool = TavilySearchResults(
    description="Use this tool to search for information about the Bauls of Bengal on the web.",
    max_results=3,
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

tools = [pinecone_tool, tavily_tool]

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Helper Functions ---
current_hour = datetime.now().hour
def get_day_period():
    if 0 <= current_hour < 12:
        return "Morning"
    elif 12 <= current_hour < 18:
        return "Afternoon"
    elif 18 <= current_hour < 24:
        return "Evening"

assistant_baul_avatar_path = os.path.abspath(os.path.join(os.getcwd(), "materials", "baulRaga_avatar_3.png"))
assistant_baul_image_path = os.path.abspath(os.path.join(os.getcwd(), "materials", "baulRaga_image.jpeg"))

def restricted_agent_executor(query: dict):
    try:
        result = agent_executor(query)

        tavily_results = []
        if "intermediate_steps" in result:
            for tool_result in result["intermediate_steps"]:
                if tool_result[0].name == "tavily_search":
                    tavily_results = tool_result[1]
                    break

            if isinstance(tavily_results, dict) and "organic_results" in tavily_results:
                tavily_results = tavily_results["organic_results"]
            elif isinstance(tavily_results, list):
                pass
            else:
                tavily_results = []
                print("Tavily results are in unexpected format:", tavily_results)
        else:
            print("No intermediate steps found. LLM might have answered directly.")

        return result["output"], tavily_results

    except Exception as e:
        print(f"Agent execution error: {e}")
        return "An error occurred during agent execution.", []

st.markdown(
    f"""
    <style>
        body {{
        color: unset;
        background-color:unset;
    }}
    [data-testid="stApp"] {{  /* Target the stApp element */ 
    background-color:rgba(155, 235, 151, 0.74);

    }}
    .stSidebar {{
        background-color:#424243;
    }}

    /* It's still a good idea to have this padding for the main content */
    .main .block-container {{ 
      padding-top: 50px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit App ---
st.title('The BaulRaga Assistant Bot')

with st.sidebar:
    st.image(assistant_baul_image_path, use_container_width=True)
    user_chat_input = st.chat_input()

if "messages" not in st.session_state:
    st.session_state.messages = [{
                                    "role": "assistant", 
                                    "content": f"""Hey! Good {get_day_period()}. My name is BaulRaga and i'm a cultural assistant chatbot. And what is your name?"""
                                }]

for message in st.session_state.messages:
    avatar = assistant_baul_avatar_path if message["role"] == "assistant" else "user"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

if user_input := user_chat_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # with st.chat_message("user", avatar="user"):
    with st.chat_message("user",  avatar="user"):
        st.write(user_input)

    # with st.chat_message("assistant", avatar=user_baul_avatar_path):
    with st.chat_message("assistant", avatar=assistant_baul_avatar_path):
        with st.spinner("Thinking..."):
            response_placeholder = st.empty()
            full_response = ""

            try:
                # Include chat history in the prompt
                chat_history = []
                for msg in st.session_state.messages:
                    chat_history.append((msg["role"], msg["content"]))

                agent_response, tavily_results = restricted_agent_executor({"input": user_input, "chat_history": chat_history}) #Include chat history

                image_links = []
                if tavily_results:
                    for item in tavily_results:
                        if item.get("images"):
                            for image in item.get("images"):
                                image_links.append(image.get("url"))

                if image_links:
                    st.image(image_links, width=200)
                    full_response += "\n\nImages from search:\n" + "\n".join(image_links)

                for chunk in agent_response.split():
                    full_response += chunk + " "
                    response_placeholder.write(full_response)
                    time.sleep(0.05)

            except Exception as e:
                print(f"Streamlit error: {e}")
                st.write("An error occurred. Please try again.")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
