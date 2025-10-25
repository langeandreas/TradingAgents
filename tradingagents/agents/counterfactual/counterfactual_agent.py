def create_counterfactual_agent(llm, memory):
    def counterfactual_node(state):
        market_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        
        curr_situation = f"{market_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        
        past_memory_str = ""
        for rec in past_memories:
            past_memory_str += rec["recommendation"] + "\n\n"
        
        prompt = f"""You are a Counterfactual Scenario Analyst. Generate 4 concise market scenarios.

For each scenario provide (MAX 50 words each):
1. Scenario name and description  
2. Probability (0.1-0.4)
3. Stock impact prediction
4. Investment implication

Required scenarios:
- Bull Market: Strong growth, low rates
- Bear Market: Economic slowdown, volatility  
- Sideways: Range-bound trading
- Black Swan: Major unexpected events

KEEP TOTAL RESPONSE UNDER 400 WORDS.

Market Report: {market_report[:300]}
Sentiment: {sentiment_report[:200]}
News: {news_report[:200]}
Fundamentals: {fundamentals_report[:300]}
Past lessons: {past_memory_str[:200]}

Provide concise, actionable scenarios."""
        
        response = llm.invoke(prompt)
        
        print("Counterfactual scenario analysis generated.")
        return {"counterfactual_analysis": f"Counterfactual Analyst: {response.content}"}
    
    return counterfactual_node
