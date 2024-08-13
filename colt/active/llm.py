import re
from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model(model):
    return pipeline("text-generation", model=model, max_new_tokens=1000)

def generate_prompt(
        portfolio,
        pool,
        mode="max",
        observation="observations",
        header="",
):
    return header + \
        f"Given a list of molecules with associated {observation}: \n" \
        + "\n".join([str(molecule) + "\t" + str(property) for molecule, property in portfolio]) + "\n" \
        + f"Which of the following molecules would you predict to have the {mode} {observation}? \n" \
        + "\n".join(str(molecule) for molecule in pool) + "\n" \
        + "Wrap the answer in <PICK></PICK> tags."
        
def parse_tags(text):
    matches = re.findall(r"<PICK>(.*?)</PICK>", text)
    if len(matches) == 0:
        return None
    return matches[0]
        
def generate_text_with_prompt(
        model,
        prompt,
):
    model = get_model(model)
    return model(prompt)

def pick(
        portfolio,
        pool,
        model,
        mode="max",
        observation="observations",
        header="",
):
    prompt = generate_prompt(portfolio, pool, mode, observation, header)
    text = generate_text_with_prompt(model, prompt)
    return parse_tags(text[0]["generated_text"])