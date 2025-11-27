import string
import re

def remove_special_characters(text):
    cleaned_text = "".join([char for char in text if char not in string.punctuation])
    return cleaned_text

def remove_html_tags(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    return cleaned_text