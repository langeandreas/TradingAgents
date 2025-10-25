from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_news,
        ]

        system_message = (
            "You are a social media and company specific news researcher/analyst tasked with analyzing social media posts, recent company news, and public sentiment for a specific company over the past week. You will be given a company's name your objective is to write a comprehensive long report detailing your analysis, insights, and implications for traders and investors on this company's current state after looking at social media and what people are saying about that company, analyzing sentiment data of what people feel each day about the company, and looking at recent company news. Use the get_news(query, start_date, end_date) tool to search for company-specific news and social media discussions. Try to look at all sources possible from social media to sentiment to news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            +"You are a helpful AI assistant, collaborating with other assistants."
            +" Use the provided tools to progress towards answering the question."
            +"ANALYZE THE DATA - do not explain or clean it. The data is already formatted correctly. "
            +"IMPORTANT: When calling get_news, limit your search to only the past 3-4 days to avoid overwhelming data. Use a date range of maximum 4 days."
            +"Upon using the tool, you will receive structured JSON data as a response, analyze the messages and sentiment scores in it to provide insights into the company's public perception."
            +"Do not give instructions to the user on how to analyze the data, just provide the analysis."
            +"keep your analysis to 400 words maximum."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}"
                    "Generate a detailed sentiment analysis report based on social media posts and recent news about the company."
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        # If the agent is making tool calls, don't set report content yet
        if len(result.tool_calls) > 0:
            report = ""
        else:
            # No more tool calls - generate the report
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
