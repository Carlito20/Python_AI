from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

class Research(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

# Initialize the language model
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
parser = PydanticOutputParser(pydantic_object=Research)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "You are a research assistant. I can help you with your research. What topic would you like me to research?"},
        {"role": "user", "content": "{query}"},
        {"role": "assistant", "content": "{agent_scratchpad}"},
    ]
).partial(format_instructions=parser.get_format_instructions())  # Fixed method name

# Create the agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

# Define the query
query = "What is the meaning of life?"

try:
    # Invoke the agent and get the response
    raw_response = agent_executor.invoke({"query": query})

    # Print or process the response
    print(raw_response)

except Exception as e:
    print(f"An error occurred: {e}")