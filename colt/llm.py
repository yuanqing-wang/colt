import re
import torch
from typing import List
from transformers import pipeline
from functools import lru_cache, partial
from vllm import LLM, SamplingParams
from .acquisition import Acquisition
from .molecule import Molecule

@lru_cache(maxsize=1)
def get_model(model):
    # return pipeline("text-generation", model=model, max_new_tokens=100, device="cuda:0", num_return_sequences=1,)
    llm = LLM(model)
    params = SamplingParams(max_tokens=100)
    model = lambda x: llm.generate(x, params)
    return model

def generate_prompt(
        past,
        future,
        mode="max",
        observation="observations",
        header="",
):
    return header + \
        f"Given a list of molecules with associated {observation}: \n" \
        + "\n".join([str(molecule) + "\t" + str(property) for molecule, property in past]) + "\n" \
        + f"Which of the following molecules would you predict to have the {mode} {observation}? \n" \
        + "\n".join(str(idx) + "\t" + str(molecule) for idx, molecule in enumerate(future)) + "\n" \
        + "Make sure the answer is an integer indexing the options. \n" \
        + "Wrap the index of the answer in <PICK></PICK> tags."
        
def parse_tags(text):
    matches = re.findall(r"<PICK>(.*?)</PICK>", text)
    if len(matches) == 0:
        return None
    matches = [match.strip() for match in matches if match != ""]
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
        past,
        future,
        model,
        mode="max",
        observation="observations",
        header="",
        tries=10,
):
    prompt = generate_prompt(past, future, mode, observation, header)
    for _ in range(tries):
        text = generate_text_with_prompt(model, prompt)
        text = text[0].outputs[0].text
        best = parse_tags(text)
        if best is not None:
            if best.isdigit():
                best = int(best)
                if best < len(future):
                    best = future[best]
                    return best
    else:
        return None
        
def tournament(
    past,
    future,
    chunk=5,
    **kwargs,
):
    chunks = [future[i:i + chunk] for i in range(0, len(future), chunk)]
    picks = [pick(past, chunk, **kwargs) for chunk in chunks]
    best = pick(past, picks, **kwargs)
    return best
    
    
class LLMAcquisition(Acquisition):
    def __init__(self, model, tries=10):
        self.model = model
        self.tries = tries
        
    def pick(
        self,
        past: List[Molecule],
        future: List[Molecule],
    ):
        past.shuffle()
        future.shuffle()
        past = [(molecule.smiles, molecule.y) for molecule in past]
        future = [molecule.smiles for molecule in future]
        best = None
        for _ in range(self.tries):
            best = pick(past=past, future=future, model=self.model)
            if best in future:
                best = Molecule(best)
                break
        if best is None:
            best = Molecule(future[0])
        return best