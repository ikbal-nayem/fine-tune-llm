def calculateDuration(start_time, end_time) -> tuple[int, int, int]:
    elapsed_seconds = end_time - start_time

    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)

    return hours, minutes, seconds

def convertBn2En(bangla_number_string)->str:
    bangla_digits = '০১২৩৪৫৬৭৮৯'
    english_digits = '0123456789'
    conversion_map = str.maketrans(bangla_digits, english_digits)
    return bangla_number_string.translate(conversion_map)