tests = [["a"],["b"],["c"]]

for i, t in enumerate(tests[2:]):
  print(i)
  tests[i] = list(filter(lambda x: x is "b", t))

print(tests)