from functools import partial
import crem
from .generative_task import GenerativeTask

class Grow(GenerativeTask):
    """Grow a molecule."""
    implementation: lambda mol: crem.grow(
        mol, db_name="replacements02_sa2.db.gz"
    )
    
class Mutate(GenerativeTask):
    """Mutate a molecule."""
    implementation: lambda mol: crem.mutate(
        mol, db_name="replacements02_sa2.db.gz"
    )
