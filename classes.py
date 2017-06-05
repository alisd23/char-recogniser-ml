charRanges = {
  'digits': range(48, 58),
  'uppercase': range(65, 91),
  'lowercase': range(97, 123)
}

# Map of class number (index) to ascii code
classNo = 0
classToCode = {}
codeToClass = {}

def addClass(charCode):
  global classNo, classToCode, codeToClass
  classToCode[classNo] = charCode
  codeToClass[charCode] = classNo
  classNo += 1

for charCode in charRanges['digits']:
  addClass(charCode)

for charCode in charRanges['uppercase']:
  addClass(charCode)

for charCode in charRanges['lowercase']:
  addClass(charCode)
