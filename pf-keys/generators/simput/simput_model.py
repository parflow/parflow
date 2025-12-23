import click
import json
import re
import copy
from pathlib import Path
from yaml import load, Loader
from collections import defaultdict


exts = [".yaml"]

lookup_tables_by_class = dict()
lookup_attribute_by_path = dict()
lookup_domain_dependency_by_pattern = defaultdict(list)
external = dict()


#  _____                    _       _
# |_   _|__ _ __ ___  _ __ | | __ _| |_ ___  ___
#   | |/ _ \ '_ ` _ \| '_ \| |/ _` | __/ _ \/ __|
#   | |  __/ | | | | | |_) | | (_| | ||  __/\__ \
#   |_|\___|_| |_| |_| .__/|_|\__,_|\__\___||___/
#                    |_|
# ------------------------------------------------------------
# Templates
# ------------------------------------------------------------

# Const values
STRING = "AnyString"
BOOL = "BoolDomain"
DOUBLE = "DoubleValue"
ENUM = "EnumDomain"
INT = "IntValue"
MANDATORY = "MandatoryValue"

TYPES = [STRING, BOOL, DOUBLE, ENUM, INT]


# Base Model
model = {
    "output": {},
    "defaultActiveView": "Core",
    "order": [],
    "views": {},
    "definitions": {},
}


# Name Parameter
def name_param(att_name):
    parent, param_id = att_name.split("/")
    param_id = param_id.replace(".{", "").replace("}", "")
    name = [
        {
            "id": f"{param_id}_",
            "label": "Name",
            "size": 1,
            "type": "string",
            "help": f"User-defined instance from {parent} Names",
        }
    ]

    return name


# Dynamic View Base
def dyn_view(att_name, label, param_id):
    view = {
        "label": label,
        "attributes": [att_name],
        "size": -1,
        "hooks": [
            {
                "type": "copyParameterToViewName",
                "attribute": f"{att_name}.{param_id}",
            }
        ],
    }

    return view


#   ____                           _
#  / ___| ___ _ __   ___ _ __ __ _| |_ ___  _ __
# | |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
# | |_| |  __/ | | |  __/ | | (_| | || (_) | |
#  \____|\___|_| |_|\___|_|  \__,_|\__\___/|_|
#
# ------------------------------------------------------------
# Generator
# ------------------------------------------------------------


def create_view(data, fname):
    # By default treat each new file as a view
    view = model["views"]
    atts, new_views, attr_paths = check_attributes(data)
    if atts:
        model["order"].append(fname)
        view[fname] = {"label": fname, "attributes": atts}
        create_definitions(atts)

    return new_views, attr_paths


def check_attributes(data, attr_paths=None):
    # If one of the attributes of a view is a user defined key,
    # break it out into its own view instead
    attr_paths = {} if attr_paths is None else attr_paths
    atts, dyn_views = [], []
    for att in data.keys():
        keys = data[att].keys()
        children = filter_keys(keys, [".{", "_"], included=False)
        dyn_view = filter_keys(keys, [".{"], included=True)
        if children:
            atts.append(att)
            attr_paths[att] = att
        if dyn_view:
            for view in dyn_view:
                if not is_variable_table(data[att][view]):
                    dyn_views.append(f"{att}/{view}")

    return atts, dyn_views, attr_paths


def create_definitions(atts, path=None, dyn=False):
    path = atts if path is None else path
    # Add new attributes to the definition
    for att in atts:
        model["definitions"][att] = {
            "label": att.replace("_", " "),
            # Dynamic views require an extra parameter so that the
            # user can set its name.
            "parameters": name_param(path) if dyn else [],
        }


def create_dynamic_view(data, views, fname, attr_paths):
    view = model["views"]
    for path in views:
        keys = path.split("/")
        # Avoid repetitive naming, i.e. "Geom GeomInput Properties"
        name = f"{fname} {keys[0]}" if not keys[0].startswith(fname) else f"{keys[0]}"
        label = f"{name} Properties"

        att_name = label.replace(" ", "_")
        attr_paths[att_name] = path
        if lookup_attribute_by_path.get(path):
            raise ValueError("Error: Multiple attributes sharing path", path, att_name)
        lookup_attribute_by_path[path] = att_name
        model["order"].append(att_name)
        param_id = parse_key(path.split("/")[-1])
        view[att_name] = dyn_view(att_name, label, f"{param_id}_")
        create_definitions([att_name], path, dyn=True)


def find_params(att, data, prefix="", params=None, table=False):
    if child_of_dynamic_view(prefix) and not dynamic_view(att) and not table:
        return

    # Find and return all keys that require input (the model parameters)
    params = {} if params is None else params
    for token, value in data.items():
        if isinstance(value, dict):
            if token_is_leaf(value) and not general_annotation(token):
                params[f"{prefix}{token}"] = value
            elif is_variable_table(value):
                params[f"{prefix}{token}"] = {
                    "table_params": {},
                    "__class__": value["__class__"],
                    "table_label": value.get("__simput__", {}).get("table_label", None),
                    "match_condition": value.get("__simput__", {}).get(
                        "match_condition"
                    ),
                }
                find_params(
                    att,
                    value,
                    f"{prefix}{token}/",
                    params[f"{prefix}{token}"]["table_params"],
                    table=True,
                )
            else:
                if child_of_dynamic_view(token) and dynamic_view(att):
                    params[f"{parse_key(token)}_"] = value
                elif intermediate_token(value):
                    params[f"{prefix}{token}"] = value["__value__"]
                find_params(att, value, f"{prefix}{token}/", params, table)

    return params


def is_variable_table(value):
    return value.get("__simput__", {}).get("type", None) == "VariableTable"


def clean_dynamic_tokens(param_id):
    for token in param_id.split("/"):
        if parse_key(token) != token:
            param_id = param_id.replace(token, f"{parse_key(token)}_")
    return param_id


def _create_parameters(path, id, data):
    # Create the parameter and set its attributes
    label = parse_parameter_label(id)

    # '.{dynamic_token}/path' -> 'dynamic_key_/path'
    param_id = f"{path}/{id}"
    param_id = clean_dynamic_tokens(param_id)

    leaf_token = param_id.split("/")[-1]

    if general_annotation(leaf_token):
        param_id = param_id.replace(f"/{leaf_token}", "")
        label = param_id

    if leaf_token.endswith("_") and data.get("table_params") is None:
        param_id = param_id.split("/")[-1]

    param = {"id": param_id, "label": label, "size": 1}

    if MANDATORY in data.get("domains", {}).keys():
        # Indicate parameter is mandatory
        param["label"] = f'{param["label"]} (REQUIRED)'

    table_label = data.get("__simput__", {}).get("table_label")
    if table_label:
        param["label"] = table_label

    param_type = _set_type(param, data)

    default = data.get("default", None)
    if default is not None:
        # Set the default value if given
        param["default"] = [default] if param_type == ENUM else default

    # Clean up and add the help text
    help_text = data.get("help", "").strip("\n").split("] ", 1)
    param["help"] = help_text[1] if len(help_text) > 1 else help_text[0]

    if param_id.lower().endswith("filename"):
        param["domain"] = {"dynamic": True, "external": "files"}
        param["ui"] = "enum"

    # Add table config
    table = data.get("table_params")
    if table is not None:
        param["ui"] = "variable_table"
        external_key = f"VariableTableDomain/{param_id}"
        param["domain"] = {"dynamic": True, "external": external_key}
        external[external_key] = {
            "columns": data.get("table_params"),
            "variable_columns": data.get("variable_columns"),
            "table_labels": data.get("table_labels"),
            "table_order": data.get("table_order"),
        }
        param["default"] = {"rows": []}

    # Register classes and handlers for hook
    param["handlers"] = list(data.get("handlers", {}).values())
    domain_dep = data.get("domains", {}).get("EnumDomain", {}).get("locations")
    if domain_dep:
        param["domain"] = {"dynamic": True, "external": f"EnumDomain/{param['id']}"}
        for location in domain_dep:
            clean_location = location[1:]  # drop leading slash
            pattern = re.sub(r"\{.*?\}", ".*", clean_location)
            lookup_domain_dependency_by_pattern[pattern].append(param["id"])

    if data.get("__class__"):
        if lookup_tables_by_class.get(data["__class__"]):
            raise ValueError("Error: multiple tables sharing class", data["__class__"])
        lookup_tables_by_class[data["__class__"]] = {
            "variable_column_id": param["id"],
            "path": path,
            "table": table,
        }
        match_condition = data.get("match_condition")
        if match_condition:
            lookup_tables_by_class[data["__class__"]][
                "match_condition"
            ] = match_condition

    return param


def _set_type(param, data):
    # Determine and set parameter type
    domains = data.get("domains", {}).keys()
    param_type = next((key for key in domains if key in TYPES), False)
    if param_type == ENUM:
        param["ui"] = "enum"
        enum = data.get("domains", {}).get(ENUM, {})
        enum_list = enum.get("enum_list", [])
        if enum_list and all("v" in opt for opt in enum_list):
            # For now use the most recent version if more than one listed
            latest_version = list(enum_list.keys())[-1]
            enum_list = enum.get("enum_list", [])[latest_version]
        param["domain"] = {val: val for val in enum_list}
    else:
        if param_type == BOOL:
            param["type"] = "bool"
        elif param_type == DOUBLE:
            param["type"] = "double"
        elif param_type == INT:
            param["type"] = "int"
        else:
            param["type"] = "string"

    return param_type


def value_from_path(data, path):
    # Takes string path in form "path/to/value"
    return value_from_path(data[path[0]], path[1:]) if path else data


def filter_keys(keys, delimiters, included=False):
    for d in delimiters:
        keys = [k for k in keys if (d in k) is included]
    return keys


def parse_key(key):
    if not key.startswith(".{"):
        return key
    return re.findall("\{(.*)\}", key)[0]


def dynamic_token(key):
    return key.endswith("}/")


def child_of_dynamic_view(key):
    return key.startswith(".{")


def dynamic_view(att):
    return "Properties" in att


def general_annotation(key):
    return key.startswith("_")


def intermediate_token(item):
    return "__value__" in item.keys()


def token_is_leaf(item):
    return isinstance(item, dict) and "help" in item.keys()


def parse_parameter_label(id):
    # '.{dynamic_token}/path' -> 'Dynamic Path'
    label = []
    for value in id.split("/"):
        item = parse_key(value)
        if item[0].islower():
            item = item.title()
        label.append(item.rsplit("_", 1)[0])
    return (" ").join(label)


def create_parameters(data, attr_paths):
    # Use the path associated with each attribute to find all of
    # the attribute's parameters
    for att, path in attr_paths.items():
        processed_params = []
        values = value_from_path(data, path.split("/"))
        param_data = find_params(att, values)
        for id, p_data in param_data.items():
            if p_data.get("table_params"):
                processed_params.extend(make_tables(path, id, p_data))
            else:
                processed_params.append(_create_parameters(path, id, p_data))
        model["definitions"][att]["parameters"].extend(processed_params)


def make_tables(path, id, data, dynamic_tokens=None, table_labels=None):

    # Configure table details
    table_order = {}
    if table_labels is None:
        table_labels = {}
    if data.get("table_label"):
        simput_id = f"{path}/{clean_dynamic_tokens(id)}"
        table_labels[simput_id] = data.get("table_label")

    # Collect parameters of children
    tables = []
    table_params = []
    if dynamic_tokens is None:
        dynamic_tokens = []
    dynamic_tokens.append(id)
    for variable_column_id, table_p in data.get("table_params").items():
        if (
            table_p.get("table_params") is not None
        ):  # If child is a table, make it a sibling table
            tables.extend(
                make_tables(
                    path,
                    variable_column_id,
                    table_p,
                    dynamic_tokens.copy(),
                    table_labels,
                )
            )
        else:
            table_param = _create_parameters(path, variable_column_id, table_p)
            table_params.append(table_param)

            # Add details from children to table
            simput_id = f"{path}/{clean_dynamic_tokens(variable_column_id)}"
            label = table_p.get("__simput__", {}).get("table_label", None)
            order = table_p.get("__simput__", {}).get("table_column_order", None)
            if label:
                table_labels[simput_id] = label
            if order:
                table_order[simput_id] = order

    data_with_children = {
        "__class__": data.get("__class__"),
        "table_labels": table_labels,
        "table_order": table_order,
        "table_params": table_params,
        "variable_columns": {
            f"{clean_dynamic_tokens(path)}/{clean_dynamic_tokens(token)}": []
            for token in dynamic_tokens
        },
        "match_condition": data.get("match_condition"),
    }

    # Fewer variables go first
    tables.insert(0, _create_parameters(path, id, data_with_children))

    return tables


def attach_direct_hooks(view, param, attr, hooks_by_views):
    if param.get("handlers"):
        hooks_by_views[view].extend(_attach_hooks(param, attr))


def attach_nested_hooks(view, param, attr, hooks_by_views):
    external_key = f'VariableTableDomain/{param["id"]}'
    for nested_param in external.get(external_key, {}).get("columns", []):
        attach_direct_hooks(view, nested_param, attr, hooks_by_views)


def attach_hooks():
    hooks_by_views = defaultdict(list)
    for view in model["views"]:
        for attr in model["views"][view]["attributes"]:
            for param in model["definitions"][attr]["parameters"]:
                attach_direct_hooks(view, param, attr, hooks_by_views)
                attach_nested_hooks(view, param, attr, hooks_by_views)

    for view, hooks in hooks_by_views.items():
        for hook in hooks:
            hook.update({"names_view": view})
        model["views"][view]["hooks"] = model["views"][view].get("hooks", [])
        model["views"][view]["hooks"].extend(hooks)


def _attach_hooks(param, attr):
    hooks = []
    for handler in param["handlers"]:
        is_hook = handler["type"] == "ChildrenHandler"
        has_class = "class_name" in handler.keys()
        if is_hook and has_class:
            attach_table_from_names_hook(hooks, handler, param, attr)

        for pattern, enums in lookup_domain_dependency_by_pattern.items():
            if re.search(pattern, param["id"]):
                attach_domain_from_locations_hook(hooks, param, attr, enums)
    return hooks


def attach_domain_from_locations_hook(hooks, param, attr, dependant_domains):
    hooks.append(
        {
            "source_id": param["id"],
            "type": "ReadLocationsWriteDomainHook",
            "dependant_domains": dependant_domains,
        }
    )


def attach_table_from_names_hook(hooks, handler, param, attr):
    target = lookup_tables_by_class.get(handler["class_name"])
    if target and target.get("table") is not None:
        dynamic_attribute = lookup_attribute_by_path.get(target["path"])
        table_attr = dynamic_attribute or target["path"].split("/")[0]
        hook = {
            "table_attr": table_attr,
            "variable_column_id": target.get("variable_column_id"),
            "match_condition": target.get("match_condition"),
            "names_id": param["id"],
            "names_attr": attr,
            "type": "ReadNamesWriteTableHook",
        }
        hooks.append(hook)


@click.command()
@click.option(
    "-o",
    "--output",
    default=".",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    help="The directory to output the model file to. If no output "
    + "is provided the file will be created in the current directory.",
)
@click.option(
    "-d",
    "--directory",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help="The directory of definition files.",
)
@click.option(
    "-f",
    "--file",
    default=None,
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="A definition file to use.",
)
@click.option(
    "--include-wells",
    default=False,
    help="Whether to include support for Wells. This is complicated and left out by default.",
)
@click.option(
    "--include-clm",
    default=False,
    help="Whether to include support for CLM. This is complicated and left out by default",
)
def cli(output, directory, file, include_wells, include_clm):
    """Accepts a single file, list of files, or directory name."""
    files = Path(directory).iterdir() if directory else [Path(f) for f in file]

    for f in files:
        # Create the views, attributes, and parameters needed
        # for each file
        fname = f.stem.capitalize()

        # Wells are removed by default
        # See --include-wells
        if fname != "Wells" or include_wells:
            with open(f) as value:
                data = load(value, Loader=Loader)

                # CLM is removed by default
                # See --include-clm
                if fname == "Solver" and not include_clm:
                    data["Solver"].pop("CLM")
                if fname == "Netcdf" and not include_clm:
                    data["NetCDF"].pop("WriteCLM")
                    data["NetCDF"].pop("CLMNumStepsPerFile")

                # Create views, attributes, and parameters
                new_views, attr_paths = create_view(data, fname)
                if new_views:
                    create_dynamic_view(data, new_views, fname, attr_paths)
                create_parameters(data, attr_paths)

    # Check for handlers which get hooks after all params set
    attach_hooks()

    # For now core is first
    order = sorted(model["order"])
    if "Core" in model["order"]:
        order.insert(0, order.pop(order.index("Core")))
    model["order"] = order

    model["external"] = external

    with open(f"{output}/model.json", "w", encoding="utf8") as f:
        json.dump(model, f, ensure_ascii=False)


if __name__ == "__main__":
    cli()
