from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
#model = ChatOpenAI(model="gpt-4o")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: str(x).upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(str(x).split())}\n{x}")
replace_letter = RunnableLambda(lambda x: str(x).upper().replace('LAW','KAR'))

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | replace_letter | uppercase_output | count_words

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 1})

# Output
print(result)
