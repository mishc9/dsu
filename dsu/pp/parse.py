import json

__doc__ = "Parsing of pandas_profiler output"


def report_to_json(report):
    return report.to_json()


def report_to_dict(report):
    json_report = report_to_json(report)
    dict_report = json.loads(json_report)
    return dict_report


def get_columns_by_type(dict_report):
    """
    Extract warnings and group them by type.
    returnst: lists - missing, high_cardinality, uniform, zeros, unique, other
    """
    missing = []
    high_cardinality = []
    uniform = []
    zeros = []
    unique = []
    other = []

    messages = dict_report.get('messages')
    for msg in messages:
        msg_val = msg['__Message__']
        error, *_, colname = msg_val.split(' ')
        error = error[1:-1]
        if error == 'MISSING':
            missing.append(colname)
        elif error == 'HIGH_CARDINALITY':
            high_cardinality.append(colname)
        elif error == 'UNIFORM':
            uniform.append(colname)
        elif error == 'ZEROS':
            zeros.append(colname)
        elif error == 'UNIQUE':
            unique.append(colname)
        else:
            other.append(colname)
    return missing, high_cardinality, uniform, zeros, unique, other


def get_variable_groups(report):
    """
    Group variables by type
    returns: lists - numeric_vars, cat_vars, bool_vars
    """
    bool_vars = []
    cat_vars = []
    numeric_vars = []
    variables = report['variables']
    for var_name, var in variables.items():
        var_type = var['type']['__Variable__']
        if var_type == 'Variable.TYPE_CAT':
            cat_vars.append(var_name)
        elif var_type == 'Variable.TYPE_BOOL':
            bool_vars.append(var_name)
        elif var_type == 'Variable.TYPE_NUM':
            numeric_vars.append(var_name)
        else:
            print('unknown type')
    return numeric_vars, cat_vars, bool_vars
