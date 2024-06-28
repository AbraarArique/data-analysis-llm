from langchain_community.utilities import SQLDatabase
from datetime import datetime
from typing import List
from langchain_openai import ChatOpenAI
import json
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
import os
from chronos import chronos_prediction

# Set up database and LLM
db = SQLDatabase.from_uri("sqlite:///zillow.db")

api_key = os.environ["OPENAI_API_KEY"]
model = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=api_key)

# Create SQL tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)
sql_tools = toolkit.get_tools()

sql_tools[0].name = "query_sql_database_tool"
sql_tools[1].name = "info_sql_database_tool"
sql_tools[2].name = "list_sql_database_tool"
sql_tools[3].name = "query_sql_checker_tool"


@tool
def get_current_datetime(current: bool):
    """Get current date and time in ISO format"""
    return datetime.now().isoformat()


@tool
def get_time_series_prediction(
    historical_values: List[float], number_of_values_to_predict: int
):
    """Use this tool to generate possible future predictions based on past time series data.
    Provide a list of numbers as 'historical_values', and specify how many future values to predict in 'number_of_values_to_predict'
    This tool returns the predicted list of numbers representing median trends/forecasts. It'll also output a visual graph.
    """
    pred = chronos_prediction(historical_values, number_of_values_to_predict)
    return json.dumps(pred.tolist())


# Store tools in hash for calling
toolsHash = {
    "query_sql_database_tool": sql_tools[0],
    "info_sql_database_tool": sql_tools[1],
    "list_sql_database_tool": sql_tools[2],
    "query_sql_checker_tool": sql_tools[3],
    "get_current_datetime": get_current_datetime,
    "get_time_series_prediction": get_time_series_prediction,
}

model = model.bind_tools([*sql_tools, get_current_datetime, get_time_series_prediction])

# Provide clear instructions
system_prompt = """You are an assistant designed to help with business and data analysis.
If the user asks for data you don't have, use the provided tools/functions to interact with a database; follow these steps:

1. First, you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step

2. Then query the schema of the most relevant tables

3. Create a syntactically correct SQLite query

4. You MUST use the tool to check/validate your query syntax before executing it. If you get an error while executing a query, rewrite the query and try again

5. Run the query, look at the results, and only use this returned information to construct your final answer

Guidelines:

Do not use the 'multi_tool_use.parallel' tool, call each tool individually.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

template = ChatPromptTemplate.from_messages(
    [SystemMessage(system_prompt), MessagesPlaceholder(variable_name="messages")]
)
