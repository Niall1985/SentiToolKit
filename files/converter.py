import json
with open('neu_dup.txt', 'r') as file:
    lines = file.readlines()
formatted_reviews = []
for line in lines:
    review = line.strip()
    formatted_reviews.append({"review": review, "sentiment": "neutral"})
json_data = json.dumps(formatted_reviews, indent=4)
with open('neutral_reviews.json', 'w') as json_file:
    json_file.write(json_data)
