import numpy.random as rnd


def argmax(f, xs, *, every=False, random=False):
    if every and random:
        raise ValueError('Arguments `every` and `random` can not both be true.')

    if not every and not random:
        return max(xs, key=f)

    fmax = None
    for x in xs:
        fx = f(x)
        if fmax is None or fx > fmax:
            xmaxs, fmax = [], fx
        if fmax == fx:
            xmaxs.append(x)

    if every:
        return xmaxs

    # random
    xi = rnd.choice(len(xmaxs))
    return xmaxs[xi]
