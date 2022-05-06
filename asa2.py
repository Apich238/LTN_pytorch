import torch


# draft of the API
class Theory(torch.nn.Module):
    def __init__(self):
        pass

    def domain(self, name, dtype, shape: tuple):
        pass

    def variable(self, name, domain, precondition=None):
        pass

    def constant(self, name, domain, grounding):
        pass

    def function(self, name, domain_in, domain_out, grounding, trainable):
        pass

    def predicate(self, name, domain_in, grounding, trainable):
        pass

    def axiom(self, formula):
        pass

    def parameters(self, **kwargs):
        pass

    def forward(self, **vars):
        SatAgg = 0
        Loss = 1 - SatAgg
        raise NotImplementedError("")
        return SatAgg, Loss

    def EvaluateFormula(self, formula, **vars):
        pass


def binary_classification_example():
    K = Theory()

    # domains
    K.domain('point', 'fp32', (2,))
    # vars
    K.variable('x', 'point', )
    K.variable('x_pos', 'point', precondition)
    K.variable('x_neg', 'point', precondition)
    # constants
    # K.constant(name, domain, grounding)
    # functions
    # K.function(name, domain_in, domain_out, grounding)
    # predicates
    K.predicate('A', ['point'], grounding, True)
    # axioms
    K.axiom('FORALL(x_pos):A(x_pos)')
    K.axiom('FORALL(x_neg):!A(x_neg)')

    # optimizer
    opt = torch.optim.Adam(K.parameters())
    # dataset
    ## generate data

    ## dataset
    from torch.utils.data import Dataset
    train_dataset = None
    test_dataset = None
    ##dataloader

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)
    ## objective function and loss function
    sat, loss = K(x=[], x_pos=[], x_neg=[])
    # quering
    # 1.
    K.evaluate_formula('A(x)', x=[])
    # 2.
    y = K.A(x=[])
    # 3.
