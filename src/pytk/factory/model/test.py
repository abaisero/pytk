import pytk.factory.model as fmodel
import pytk.factory as factory


if __name__ == '__main__':

    f1 = factory.FactoryN(2)
    f2 = factory.FactoryN(3)
    f3 = factory.FactoryN(2)
    f4 = factory.FactoryN(3)

    model = fmodel.Softmax(f1, f2, cond=(f3, f4))

    i1 = f1.item(0)
    i2 = f2.item(1)
    i3 = f3.item(1)
    i4 = f4.item(2)

    print(model.prefs(i1, i2))
    print(model.prefs(i1, i2, ..., ...))

    print(model.logprobs(i1, i2))
    print(model.logprobs(i1, i2, ..., ...))

    print(model.probs(i1, i2))
    print(model.probs(i1, i2, ..., ...))

    print(model.probs(..., ..., ..., ...))  # NOT A PROBLEM!!!!!  THIS ISNT A CONDITIONAL DISTRIBUTION!!
    print(model.probs(i1, i2, i3, i4))
    print(model.probs(..., i2, i3, i4))

    # print(model.phi(i1, i2, i3, i4))
    # print(model.dprefs(i1, i2, i3, i4))
    print(model.dlogprobs(i1, i2, i3, i4))
