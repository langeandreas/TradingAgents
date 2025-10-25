def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

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

        # Check if this is responding to a bull argument
        debate_count = investment_debate_state.get("count", 0)
        bull_argument = current_response if current_response.startswith("Bull") else ""

        if bull_argument:
            focus_instruction = f"CRITICAL: The bull analyst just claimed: '{bull_argument[:200]}...' Counter with NEW risk factors that haven't been discussed yet. Do NOT repeat previous bear points from history: {bear_history[:300]}. Find fresh concerns and weaknesses."
        else:
            focus_instruction = "Present your initial bear case with 3-4 specific risk factors and concerns. Focus on concrete data and avoid generic warnings."

        prompt = f"""You are a Bear Analyst arguing against investing in this stock. {focus_instruction}

STRICT REQUIREMENTS:
1. DO NOT repeat arguments already made in your previous statements: {bear_history}
2. If responding to bull analyst, directly quote and challenge their specific claims
3. Introduce NEW risks, data points, or concerns not yet discussed
4. Keep response focused and under 200 words
5. End with one specific reason why the investment is risky now

DEBATE CONTEXT:
- Current debate round: {debate_count + 1}
- Previous bear arguments made: {bear_history[:400] if bear_history else "None yet"}
- Bull's latest claim: {bull_argument[:400] if bull_argument else "None yet"}

DATA SOURCES (use selectively to find NEW risks):
Market research: {market_research_report[:500]}...
Fundamentals: {fundamentals_report[:500]}...
Sentiment: {sentiment_report[:300]}...
News: {news_report[:300]}...
Counterfactual scenarios: {counterfactual_analysis[:400] if counterfactual_analysis else "None available"}...

COUNTERFACTUAL INSTRUCTION: Use the scenario analysis to identify which future conditions would make this investment particularly risky. Reference specific scenarios and their probabilities in your argument.

Deliver a focused, novel bear argument that exposes new risks or challenges the bull's latest claims.
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }
        print("Bear argument generated.")
        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
