def calculateDuration(start_time, end_time) -> tuple[int, int, int]:
    elapsed_seconds = end_time - start_time

    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)

    return hours, minutes, seconds
