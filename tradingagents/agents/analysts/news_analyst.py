from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + "You are a helpful AI assistant, collaborating with other assistants."
            + " Use the provided tools to progress towards answering the question."
            + "Analyze the news data you retrieve to generate insights that may impact the stock market and trading decisions."
            + "Do not give instructions to the user on how to analyze the data, just provide the analysis."
            + "Provide only a detailed report, do not provide instructions or explanations about your process."
            + "keep your analysis to 400 words maximum."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}"
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
            
        print("News Analyst Report generated")
        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
