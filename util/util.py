import re


def get_url(string_input):
    return re.findall(r'(https?://\S+)', string_input)