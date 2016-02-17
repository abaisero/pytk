import picklejar

jar = picklejar.Jar()

print(jar.label)
jar.labels.append('OK')
print jar.label
jar.labels.append('OK')
print jar.label
# jar.seal('PICKLE')
print jar.unseal()
