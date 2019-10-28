from datetime import timedelta



def get_str_delta(delta: timedelta):
    """
    Converting timedelta to pandas format
    :param delta:
    :return:
    """
    seconds = delta.seconds
    if seconds >= 3600 * 24:
        raise ValueError("Timedelta should be shorter than a day")
    hours = seconds // 3600
    minutes_seconds = seconds % 3600
    minutes = minutes_seconds // 60
    seconds = minutes_seconds % 60
    # should I use templates?
    seconds_str = f"{seconds}S" if seconds else ""
    minutes_str = f"{minutes}M" if minutes else ""
    hours_str = f"{hours}H" if hours else ""
    str_delta = ''.join([hours_str, minutes_str, seconds_str])
    if not str_delta:
        raise ValueError("Empty str_delta means infinite frequency, should change time_delta")
    return str_delta
