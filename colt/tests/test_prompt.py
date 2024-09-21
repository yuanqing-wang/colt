def test_prompt():
    from colt.active.llm import generate_prompt
    portfolio = [("C", 1), ("CC", 2)]
    pool = ["CCC", "CCCC"]
    prompt = generate_prompt(portfolio, pool)
    print(prompt)
    
def test_pick():
    from colt.active.llm import pick
    portfolio = [("C", 1), ("CC", 2)]
    pool = ["CCC", "CCCC"]
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    result = pick(portfolio, pool, model)
    print(result)
    
    