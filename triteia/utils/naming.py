def hf_name_to_path_friendly_name(name):
    # replace "/" with "."
    # in all lower letters
    return name.replace("/", ".").lower()