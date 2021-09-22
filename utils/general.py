from clearml import Task


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


def get_args_from_clearml(task=None, project_name=None, task_name=None):
    if not task:
        if not project_name or not task_name:
            raise Exception('You must provide both project_name adn task_name')
        task = Task.get_task(project_name=project_name, task_name=task_name)

    args = {}
    for key, value in task.get_parameters().items():
        key_type, key_name = key.split('/')
        if key_type == 'Args':
            args[key_name] = value
    return args
