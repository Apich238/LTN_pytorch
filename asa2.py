
# draft of the API
class Theory:
    def __init__(self):
        pass

K=Theory()

# domains
K.domain(name,dtype,shape)
# vars
K.variable(name,domain,precondition:predicate)
# constants
K.constant(name,domain,grounding)
# functions
K.function(name,domain_in,domain_out,grounding)
# predicates
K.predicate(name,domain_in,grounding)
# axioms
K.axiom(predicate_formula)
# groundings
## defined at declaration
# learning

## optimizer
opt=Adam(Theory.weights())
## dataset
dataldr=DataLoader()
## objective function and loss function
sat=K.SatAgg(var1=var1,var2=var2,...)
loss=K.Loss(var1=var1,var2=var2,...)#1-K.SatAgg
# quering
K.evaluate_formula(formula,vars_values)
# reasoning