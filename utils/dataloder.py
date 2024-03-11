def char_load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
        
    # all unique chars
    chars = sorted(list(set(data)))
    return data, chars