55.# Generate Acronym from a Phrase
def generate_acronym(phrase):
    return ''.join(word[0].upper() for word in phrase.split())