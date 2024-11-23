import re
from typing import List
from colt.bayesian.bnn import NUM_SAMPLES
from vllm import LLM, SamplingParams
from ..llm import get_model
from ..molecule import Molecule
from .acquisition import (
    Acquisition,
    expected_improvement,
    probability_of_improvement,
)
import torch

def get_model(model):
    # return pipeline("text-generation", model=model, max_new_tokens=100, device="cuda:0", num_return_sequences=1,)
    # llm = LLM(model, quantization="fp8")
    llm = LLM(model, quantization="fp8", trust_remote_code=True)
    params = SamplingParams(max_tokens=50, n=NUM_SAMPLES)
    model = lambda x: llm.generate(x, params)
    return model

def generate_prompt(
        past,
        future,
        observation="observations",
        header="",
):
    return header + \
        f"Given a list of molecules with associated {observation}: \n" \
        + "\n".join([str(molecule) + "\t" + str(property) for molecule, property in zip(past[0], past[1])]) + "\n" \
        + f"What is the {observation} of the molecule {future}?" + "\n" \
        + "Make sure the answer is a float." + "\n" \
        + "Wrap the answer in <ANSWER></ANSWER> tags."
        
def parse_tags(text):
    matches = re.findall(r"<ANSWER>(.*?)</ANSWER>", text)
    if len(matches) == 0:
        return None
    matches = [match.strip() for match in matches if match != ""]
    if len(matches) == 0:
        return None
    return matches[0]

def is_float(string):
    return string.replace('.','',1).isdigit()

def predict(
    past,
    future,
    model,
    header="",
    observations="observations",
    tries=10,
):    
    answers = [[] for _ in range(len(future))]
    for idx, _future in enumerate(future):
        prompt = generate_prompt(past, _future, header=header, observation=observations)
        for _ in range(tries):
            text = model(prompt)[0]
            for output in text.outputs:
                answer = parse_tags(output.text)
                if answer is not None and len(answer) > 0 and is_float(answer):
                    try:
                        answers[idx].append(float(answer))
                    except:
                        pass
            if len(answers[idx]) >= NUM_SAMPLES:
                break
        answers[idx] = answers[idx][:NUM_SAMPLES]
        answers[idx] += [0] * (NUM_SAMPLES - len(answers[idx]))
    return answers

class BayesianLLMAcquisition(Acquisition):
    def __init__(
            self, 
            model,
            acquisition_function: callable = expected_improvement,
            n_samples=8,
            observation="observations",
        ):
        super().__init__()
        self.model = get_model(model)
        self.acquisition_function = acquisition_function
        self.n_samples=n_samples
        self.observation = observation
        
    def pick(
        self,
        past: List[Molecule],
        future: List[Molecule],
    ):
        future = [molecule.smiles for molecule in future]
        past = ([molecule.smiles for molecule in past], [molecule.y for molecule in past])
        scores = predict(past, future, self.model, observations=self.observation)
        scores = torch.tensor(scores).transpose(0, 1)
        scores = self.acquisition_function(scores).flatten()
        pick = torch.argmax(scores)
        pick = future[pick]
        pick = Molecule(pick)
        return pick
        