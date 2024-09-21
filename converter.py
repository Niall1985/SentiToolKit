import json
with open('neg_dup.txt', 'r') as file:
    lines = file.readlines()
formatted_reviews = []
for line in lines:
    review = line.strip()
    formatted_reviews.append({"review": review, "sentiment": "negative"})
json_data = json.dumps(formatted_reviews, indent=4)
with open('negative_reviews.json', 'w') as json_file:
    json_file.write(json_data)
