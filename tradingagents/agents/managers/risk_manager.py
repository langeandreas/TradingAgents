def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]
        counterfactual_analysis = state.get("counterfactual_analysis", "No counterfactual analysis available.")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for rec in past_memories:
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts—Risky, Neutral, and Safe/Conservative—and determine the best course of action for the trader. Your decision must result in a clear recommendation: Buy, Sell, or Hold. Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid. Strive for clarity and decisiveness.

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.
2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
3. **Scenario Analysis**: Analyze the counterfactual scenarios and incorporate the risk analysts' suggestions for each scenario.
4. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and adjust it based on the analysts' insights and scenario considerations.
5. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision you are making now to make sure you don't make a wrong BUY/SELL/HOLD call that loses money.

Deliverables (COMPLETE ALL SECTIONS):
- A clear and actionable recommendation: Buy, Sell, or Hold.
- Detailed reasoning anchored in the debate and past reflections.
- **Scenario Section**: Detail each counterfactual scenario with risk analysts' suggestions for handling each situation.
- **Complete the entire report** including refined trader's plan and lessons from past mistakes.

---

**Counterfactual Scenarios:**
{counterfactual_analysis}

---

**Analysts Debate History:**  
{history}

---

**SCENARIO ANALYSIS REQUIREMENT:**
Create a detailed scenario section that:
1. Lists each scenario from the counterfactual analysis
2. For each scenario, synthesize what the Risky, Neutral, and Safe analysts would recommend
3. Provide your risk management perspective on how to handle each scenario
4. Consider how each scenario affects the overall investment decision

Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes.

**IMPORTANT: Complete ALL sections including the full scenario analysis, refined trader's plan, lessons from past mistakes, and final actionable recommendation. Do not truncate your response.**"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
