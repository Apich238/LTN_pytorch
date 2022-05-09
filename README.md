# ltn_pytorch

Implementation of Logic Tensor Network with pytorch. 

Trying to follow 10.1016/j.artint.2021.103649

# Features to implement

- parsing axioms (str with real logic -> torch model - like object)
- domain control (variables shapes checking, type casting)
- constants
- choice of fuzzy logic (T-norm, conorm)
- variables values cross product
- Diag operator
- aggregation operators
- Guarded quantifiers
- allowing not torch.nn.Module to be predicates of functions
- calculating SatAgg