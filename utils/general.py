def parse_generic_args(args: str) -> dict:
    """
    Parses a string of groups arguments that comes from the command line
    :param args: string that contains the arguments. For instance: lr=1e-3,momentum=0.1
    """
    result = {}
    for group in args.split(','):
        key, value = group.split('=')
        result[key] = eval(value)
    return result
