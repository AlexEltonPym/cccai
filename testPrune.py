
import itertools


def prune(compare):
  toPrune = set(test[0]).issubset(compare[0]) and test[1] > compare[1]
  print(toPrune)
  return not(toPrune)



tests = [
  [[["a"], 0.9], [["b"], 0.2], [["c"], 0.1]],
  [[["a","b"], 0.3], [["b","c"], 0.4], [["a","c"], 0.7]],
  [[["a","b","c"], 0.1], [["d","e","f"],0.01]]
]


for currentRowMax in range(len(tests)):
  for  testRow in tests[:currentRowMax]:
    for test in testRow:
      for i, compareRow in enumerate(tests[currentRowMax:]):
        tests[currentRowMax+i] = list(filter(prune, compareRow))

print(len(tests))
for test in itertools.chain.from_iterable(tests):
  print(test)


