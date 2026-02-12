92.# Validate Credit Card Numbers (Luhn Algorithm)
def validate_credit_card(number):
    digits = [int(d) for d in str(number)]
    checksum = 0

    for i in range(len(digits)-2, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9

    checksum = sum(digits)
    return checksum % 10 == 0