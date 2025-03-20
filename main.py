from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import ChatOutputParser, PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

class research(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
parser=PydanticOutputParser(pydantic_object=research)

prompt = ChatPromptTemplate.message(
    [
        (
        "system", "You are a research assistant. I can help you with your research. What topic would you like me to research?",
        
        ),
        
        ( "placeholder", "{chat_history}" ),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
         
    ]
).partial(format_intsructions=parser.get_formats_instructions())



agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
    
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "What is the meaning of life?"})
    
# response = llm.invoke("What is the meaning of life?")
# print(response)
