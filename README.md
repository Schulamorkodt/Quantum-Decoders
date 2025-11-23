# quantum-decoder

A collection of quantum error correction utilities, Stim sampling circuits,
LDPC-style decoders, BP-SP decoders, and Monte Carlo simulations.


## Contents
pip install -e .
- GF(2) linear algebra tools  
- CSS logical operator extraction  
- Circulant matrix constructors  
- Stim bicycle circuit builder  
- Detector error model → check matrix tools  
- BP-SP decoder implementation  
- BPLSD comparison experiment  
- Monte-Carlo logical error rate plots  

All code is located in:  
quantum_decoder/code.py

## Example Run

Inside `example_run.py`:

```python
from quantum_decoder.code import *

# Example usage
ps, my_rates, ldpc_rates = monte_carlo_bp_sd_vs_bplds_simulation(
    H_X, H_Z,
    failures=20,
    max_iters=50,
    clip=20.0
)
