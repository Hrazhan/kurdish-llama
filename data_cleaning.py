import json
import sys

def remove_same_words(json_file_path, new_file):
    with open(json_file_path) as f:
        data = json.load(f)
    new_data = []
    count = 0
    for record in data:
        keep_record = True
        for key, value in record.items():
            if len(value.split()) < 3:
                continue
            words = value.split()
            for i in range(2, len(words)):
                if words[i] == words[i-1] == words[i-2]:
                    count += 1
                    keep_record = False
                    break
            if not keep_record:
                break
        if keep_record:
            new_data.append(record)
    with open(new_file, 'w') as f:
        json.dump(new_data, f, indent=4)
    return count

if __name__ == '__main__':
    count = remove_same_words(sys.argv[1], sys.argv[2])
    print(count)
