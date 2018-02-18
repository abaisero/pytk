import pytk.factory as factory

f1 = factory.FactoryValues([1, 2, 3])
f2 = factory.FactoryValues('abc')

f = factory.FactoryUnion(
    f1 = f1,
    f2 = f2
)

print(f.nitems)
for i in f.items:
    print(type(i), i, i.factory)

for v in f.values:
    print(type(v), v)

print(f.f1)
print(f.f2)

print(f.item(value=('f1', 1)).factory)
print(f.item(value=('f2', 'b')).factory)
