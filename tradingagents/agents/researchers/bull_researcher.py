


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        counterfactual_analysis = state.get("counterfactual_analysis", "")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for rec in past_memories:
            past_memory_str += rec["recommendation"] + "\n\n"

        # Check if this is the first round or a response to bear arguments
        debate_count = investment_debate_state.get("count", 0)
        is_first_round = debate_count == 0
        bear_argument = current_response if current_response.startswith("Bear") else ""
        
        if is_first_round:
            focus_instruction = "Present your initial bull case with 3-4 unique, compelling arguments for investment. Focus on specific data points and avoid generic statements."
        else:
            focus_instruction = f"CRITICAL: The bear analyst just argued: '{bear_argument[:200]}...' Build NEW counterarguments that haven't been discussed yet. Do NOT repeat previous bull points from history: {bull_history[:300]}. Find fresh angles and data."

        prompt = f"""You are a Bull Analyst advocating for investing in this stock. {focus_instruction}

STRICT REQUIREMENTS:
1. DO NOT repeat arguments already made in your previous statements: {bull_history}
2. If responding to bear analyst, directly quote and refute their specific claims
3. Introduce NEW evidence, data points, or perspectives not yet discussed
4. Keep response focused and under 200 words
5. End with one specific, actionable investment thesis

DEBATE CONTEXT:
- Current debate round: {debate_count + 1}
- Previous bull arguments made: {bull_history[:400] if bull_history else "None yet"}
- Bear's latest challenge: {bear_argument[:400] if bear_argument else "None yet"}

DATA SOURCES (use selectively to find NEW angles):
Market research: {market_research_report[:500]}...
Fundamentals: {fundamentals_report[:500]}...
Sentiment: {sentiment_report[:300]}...
News: {news_report[:300]}...
Counterfactual scenarios: {counterfactual_analysis[:400] if counterfactual_analysis else "None available"}...

COUNTERFACTUAL INSTRUCTION: Use the scenario analysis to identify which future conditions would make this investment particularly attractive. Reference specific scenarios and their probabilities in your argument.

Deliver a focused, novel bull argument that advances the debate with fresh insights.
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }
        print("Bull argument generated.")
        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
