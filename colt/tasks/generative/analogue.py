from functools import partial
import crem
from .generative_task import GenerativeTask

grow = partial(crem.grow, db_name="replacements02_sa2.db.gz")
mutate = partial(crem.mutate, db_name="replacements02_sa2.db.gz")

class Grow(GenerativeTask):
    """Grow a molecule."""
    implementation: lambda mol: grow(mol)
    
class Mutate(GenerativeTask):
    """Mutate a molecule."""
    implementation: lambda mol: mutate(mol)