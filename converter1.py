import json

def convert_to_lowercase(data):
    """Recursively convert all string values in the data to lowercase."""
    if isinstance(data, dict):
        return {key: convert_to_lowercase(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_lowercase(item) for item in data]
    elif isinstance(data, str):
        return data.lower()
    else:
        return data

def main(input_file, output_file):
   
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lowercase_data = convert_to_lowercase(data)

    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(lowercase_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = 'json/train.json'  
    output_file = 'json/train.json'  
    main(input_file, output_file)
