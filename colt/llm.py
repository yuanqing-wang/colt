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
    try:
        llm = LLM(model, trust_remote_code=True, quantization="fp8")
        params = SamplingParams(max_tokens=50)
        model = lambda x: llm.generate(x, params)
    except:
        model = pipeline(
            "text-generation", 
            model=model, 
            max_new_tokens=50, 
            device=0, 
            trust_remote_code=True,
            num_return_sequences=1,
            torch_dtype='float16',
            return_full_text=False,
        )
    return model

def generate_prompt(
        past,
        future,
        mode="max",
        observation="observations",
        header="",
):
    future = "\n".join([f"[{i}] {molecule}" for i, molecule in enumerate(future)])
    return header \
        + "\n".join([str(molecule) + "\t" + str(property) for molecule, property in zip(past[0], past[1])]) + "\n" \
        + f" is a list of molecules with associated {observation}: \n" \
        + f"In an active learning setting, which molecule among {future} would you pick next to {mode} {observation} quickly? \n" \
        + "Make sure the answer is an integer indexing the options. \n" \
        + "Wrap the index of the answer in [ ] brackets. Do not output anything else."
        
def parse_tags(text):
    # matches = re.findall(r"<PICK>(.*?)</PICK>", text)
    matches = re.findall(r"\[(.*?)\]", text)
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
        tries=3,
):
    prompt = generate_prompt(past, future, mode, observation, header)
    for _ in range(tries):
        text = generate_text_with_prompt(model, prompt)
        try:
            text = text[0].outputs[0].text
        except:
            text = text[0]["generated_text"]
        best = parse_tags(text)
        if best is not None:
            if best.isdigit():
                try:
                    best = int(best)
                    if best < len(future):
                        best = future[best]
                        return best
                except:
                    pass
    else:
        return None
        
def tournament(
    past,
    future,
    chunk=1000,
    **kwargs,
):
    if len(future) < chunk:
        return pick(past, future, **kwargs)
    chunks = [future[i:i + chunk] for i in range(0, len(future), chunk)]
    picks = [pick(past, chunk, **kwargs) for chunk in chunks]
    best = pick(past, picks, **kwargs)
    return best
    
class LLMAcquisition(Acquisition):
    def __init__(self, model, tries=10, header="", observation="observations"):
        self.model = model
        self.tries = tries
        self.header = header
        self.observation = observation
        
    def pick(
        self,
        past: List[Molecule],
        future: List[Molecule],
    ):
        past.shuffle()
        future.shuffle()
        past = [[molecule.smiles for molecule in past], [molecule.y for molecule in past]]
        future = [molecule.smiles for molecule in future]
        best = None
        for _ in range(self.tries):
            # best = pick(past=past, future=future, model=self.model, header=self.header, observation=self.observation)
            best = tournament(past=past, future=future, model=self.model, header=self.header, observation=self.observation)
            if best in future:
                best = Molecule(best)
                break
        if best is None:
            best = Molecule(future[0])
        return best