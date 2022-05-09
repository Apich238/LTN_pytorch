import torch

# draft of the API
class Theory(torch.nn.Module):
    def __init__(self):
        super(Theory, self).__init__()
        self.functions = {}
        self.predicates = {}
        self.trainable = set()

        self.flogic = 'min'  # others: prod, luk

    def train(self, mode: bool = True):
        for x in self.functions:
            self.functions[x].train(mode)
        for x in self.predicates:
            self.predicates[x].train(mode)

    def domain(self, name, dtype, shape: tuple):
        pass

    def variable(self, name, domain):
        pass

    def constant(self, name, domain, grounding):
        pass

    def function(self, name, domain_in, domain_out, grounding, trainable: bool = False):
        self.functions[name] = grounding
        if trainable:
            self.trainable.add(name)

    def predicate(self, name, domain_in, grounding, trainable=False):
        self.predicates[name] = grounding
        if trainable:
            self.trainable.add(name)

    def axiom(self, formula):
        pass

    def parameters(self, **kwargs):
        for fname in self.functions:
            if fname in self.trainable:
                for param in self.functions[fname].parameters(**kwargs):
                    yield param
        for pname in self.predicates:
            if pname in self.trainable:
                for param in self.predicates[pname].parameters(**kwargs):
                    yield param

    def _lossfn(self, SatAgg):
        return 1 - SatAgg

    def forward(self, **vars):
        # check vars domains

        # only for binary classification
        a = ((1 - self.predicates['A'](vars['x_pos'])) ** 2).mean()
        b = ((self.predicates['A'](vars['x_neg'])) ** 2).mean()
        SatAgg = 1 - 0.5 * (1 - (1 - a) + 1 - (1 - b)) ** 0.5
        Loss = self._lossfn(SatAgg)
        return SatAgg, Loss

    def EvaluateFormula(self, formula, **vars):
        pass