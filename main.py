import pandas as pd
from lawyer import Lawyer

judgment_criteria = """Is the following contract better for the buyer or the seller? Respond with exactly one letter. (A): Seller or (B): Buyer"""
improvement_criteria_seller = """Without adding more articles, improve the current contract by making it better for the seller."""
improvement_criteria_buyer = """Without adding more articles, improve the current contract by making it better for the buyer."""
prompt_new_contract = "Create a very short legal contract for the sale of goods (100 machines) between a buyer (West Manufacturing Inc.) and a seller (Square Machines Inc.). Return only the contract nothing else."

# Set up agents
mistral = Lawyer("mistralai/Mistral-7B-Instruct-v0.3", judgment_criteria)
phi3 = Lawyer("microsoft/Phi-3-mini-4k-instruct", judgment_criteria)
falcon = Lawyer("tiiuae/falcon-7b", judgment_criteria)
llama3 = Lawyer("meta-llama/Meta-Llama-3-8B", judgment_criteria)
llama2 = Lawyer("meta-llama/Llama-2-7b-hf", judgment_criteria)
gemma = Lawyer("google/gemma-7b", judgment_criteria)
saul = Lawyer("Equall/Saul-7B-Instruct-v1", judgment_criteria)
lawchat = Lawyer("AdaptLLM/law-chat", judgment_criteria)

agent_list = [mistral, phi3, falcon, llama3, llama2, gemma, saul, lawchat]

results = pd.DataFrame(columns=['Agent A', 'Agent B', 'Judge', 'Judgement'])

def competition(agent_a, agent_b, judges):
    print(f"\nStarting competition between {agent_a.model_name} and {agent_b.model_name}")
    
    # Agent A creates contract
    print(f"{agent_a.model_name} is generating a new contract.")
    contract = agent_a.generate_new_contract(prompt_new_contract)

    # Agent B has option to change contract
    print(f"{agent_b.model_name} is improving the contract for the buyer.")
    contract = agent_b.improve_contract(contract, improvement_criteria_buyer)
    
    # Agent A has option to change contract
    print(f"{agent_a.model_name} is improving the contract for the seller.")
    contract = agent_a.improve_contract(contract, improvement_criteria_seller)
    
    # Agent B has option to change contract
    print(f"{agent_b.model_name} is improving the contract for the buyer again.")
    contract = agent_b.improve_contract(contract, improvement_criteria_buyer)

    for judge in judges:
        print(f"{judge.model_name} is judging the final contract.")
        judgement = judge.judge_contract(contract)
        print(f"Judgement by {judge.model_name}: {judgement}")
        # save results to dataframe
        results.loc[len(results)] = [agent_a.model_name, agent_b.model_name, judge.model_name, judgement]

for agent_a in agent_list:
    for agent_b in agent_list:
        if agent_b != agent_a:
            judges = [agent for agent in agent_list if agent != agent_a and agent != agent_b]
            competition(agent_a, agent_b, judges)
            results.to_csv('results-v2.csv', index=True)

print(results)