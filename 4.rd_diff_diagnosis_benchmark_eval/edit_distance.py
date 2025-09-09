import difflib

pred = "hemolytic anemia"
gold = "autoimmune hemolytic anemia"

score = difflib.SequenceMatcher(None, pred.lower(), gold.lower()).ratio()
print(score)  # normalized similarity between 0 and 1