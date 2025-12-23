r"""
This module provides the infrastructure to load and generate the ParFlow
database structure as Python classes so IDE and runtime environment can
be used to query the help and constraints associated to each key.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import yaml

# -----------------------------------------------------------------------------

YAML_MODULES_TO_PROCESS = [
    "core",
    "geom",
    "solver",
    "wells",
    "reservoirs",
    "timing",
    "phase",
    "bconditions",
    "netcdf",
    "metadata",
    "run",
]


# -----------------------------------------------------------------------------


def is_field(key, definition):
    if key.startswith("_"):
        return False

    value = definition[key]

    if "__doc__" in value:
        return False

    return any(x in value for x in ["help", "__field__"])


# -----------------------------------------------------------------------------


def is_class(key, definition):
    if key[0] in ["_", "."]:
        return False

    value = definition[key]

    if "__doc__" in value:
        return True

    return not any(x in value for x in ["help", "__field__"])


# -----------------------------------------------------------------------------


def has_value(key, definition):
    if key.startswith("_"):
        return False

    return "__value__" in definition


# -----------------------------------------------------------------------------


def has_prefix(key, definition):
    if key.startswith("_"):
        return False

    return "__prefix__" in definition


# -----------------------------------------------------------------------------


def is_class_item(key, definition):
    return key.startswith(".")


# -----------------------------------------------------------------------------


def is_dynamic(key, definition):
    item = definition[key]
    return "__class__" in item and "__from__" in item


# -----------------------------------------------------------------------------


def json_to_python(txt):
    replacements = [
        (" true,", " True,"),
        (" false,", " False,"),
        (" null", " None"),
        (": true", ": True"),
        (": false", ": False"),
    ]
    for r in replacements:
        txt = txt.replace(*r)
    return txt


# -----------------------------------------------------------------------------


def yaml_value(yval):
    if isinstance(yval, str):
        try:
            return float(yval)
        except ValueError:
            pass

    return yval


# -----------------------------------------------------------------------------


class ValidationSummary:
    """
    This is the class to generate the Python library of ParFlow keys from the
    yaml files.
    """

    def __init__(self):
        self.has_duplicate = False
        self.class_name_count = {}
        self.class_count = 0
        self.field_count = 0

    def add_class(self, class_name):
        self.class_count += 1
        self.class_name_count.setdefault(class_name, 0)
        self.class_name_count[class_name] += 1
        return self.class_name_count[class_name] - 1

    def get_deduplicate_class_name(self, class_name, class_definition=None):
        if class_definition and "__class__" in class_definition:
            return class_definition["__class__"]
        if class_name in self.class_name_count:
            return f"{class_name}_{self.class_name_count[class_name]}"
        return class_name

    @staticmethod
    def get_class_name(class_name, class_definition=None):
        if class_definition and "__class__" in class_definition:
            return class_definition["__class__"]
        return class_name

    def add_field(self, field_name):
        self.field_count += 1

    def get_summary(self, line_separator="\n"):
        content = [
            f"Created {self.class_count} classes",
        ]
        if len(self.class_name_count) == self.class_count:
            self.has_duplicate = False
            content.append(" => No class name duplication found")
        else:
            self.has_duplicate = True
            content.append(
                f" => We found overlapping class_names ("
                f"{self.class_count - len(self.class_name_count)})"
            )
            for name in self.class_name_count:
                if self.class_name_count[name] > 1:
                    content.append(
                        f"   + {name} was defined "
                        f"{self.class_name_count[name]} times"
                    )

        content.append(f"Defined {self.field_count} fields were found")
        return line_separator.join(content)

    def print_summary(self):
        print(self.get_summary())


# -----------------------------------------------------------------------------


class PythonModule:
    """
    This class generates the Python library of ParFlow keys from the
    yaml files.
    """

    SUMMARY_INDEX = 4

    def __init__(self, indent=4):
        self.validation_summary = ValidationSummary()
        self.content = [
            "r'''",
            "--- DO NOT EDIT ---",
            "File automatically generated - any manual change will be lost",
            f"Generated on {datetime.now().strftime('%Y/%m/%d - %H:%M:%S')}",
            "",
            "'''",
            "from .core import PFDBObj, PFDBObjListNumber",
        ]
        self.str_indent = " " * indent

    def add_line(self, content=""):
        self.content.append(content)

    def add_separator(self):
        self.add_line()
        self.add_line()
        self.add_line(f"# {'-' * 78}")
        self.add_line()

    def add_class(self, class_name, class_definition):
        try:
            class_keys = class_definition.keys()
            class_members = []
            field_members = []
            class_instances = []
            class_items = []
            class_details = {}
            class_dynamic = {}
            field_with_prefix = 0
            field_prefix_value = None

            self.add_separator()

            validation_summary = self.validation_summary
            dedup_class_name = validation_summary.get_deduplicate_class_name(
                class_name, class_definition
            )
            validation_summary.add_class(
                validation_summary.get_class_name(class_name, class_definition)
            )

            inheritance = "PFDBObj"
            if "__inheritance__" in class_definition:
                inheritance = class_definition["__inheritance__"]

            self.add_line(f"class {dedup_class_name}({inheritance}):")
            if "__doc__" in class_keys:
                self.add_comment(class_definition["__doc__"], self.str_indent)

            for key, value in class_definition.items():
                if is_class(key, class_definition):
                    class_members.append(key)
                if is_field(key, class_definition):
                    field_members.append(key)
                if key == "__class_instances__":
                    class_instances = class_definition["__class_instances__"]
                if is_class_item(key, class_definition):
                    class_items.append(value)
                    if "__prefix__" in value:
                        field_with_prefix += 1
                        prefix = value["__prefix__"]
                        if field_prefix_value and field_prefix_value != prefix:
                            print(
                                "Warning: mismatched prefixes: ",
                                f"{field_prefix_value} and {prefix}",
                            )
                            print(f"Using {prefix}...")
                        field_prefix_value = prefix
                if is_dynamic(key, class_definition):
                    class_dynamic[value["__class__"]] = value["__from__"]

            if any(
                [
                    class_members,
                    field_members,
                    class_instances,
                    field_with_prefix,
                    class_dynamic,
                ]
            ) or has_prefix(class_name, class_definition):
                """
                def __init__(self, parent=None):
                  super().__init__(parent)
                  self.Topology = Topology(self)
                """
                self.add_line(f"{self.str_indent}def __init__(self, parent=None):")
                self.add_line(f"{self.str_indent * 2}super().__init__(parent)")

                if has_value(class_name, class_definition):
                    self.add_field(
                        "_value_", class_definition["__value__"], class_details
                    )

                if has_prefix(class_name, class_definition):
                    self.add_line(
                        f"{self.str_indent * 2}self._prefix_ = "
                        f"'{class_definition['__prefix__']}'"
                    )
                    if inheritance == "PFDBObjListNumber":
                        self.add_line(
                            f"{self.str_indent * 2}" f"self._details_ = " "{}"
                        )

                for instance in class_members:
                    name = validation_summary.get_deduplicate_class_name(
                        instance, class_definition[instance]
                    )
                    self.add_line(
                        f"{self.str_indent * 2}self.{instance} = " f"{name}(self)"
                    )

                for instance in class_instances:
                    # class_definition[instance]
                    self.add_class_instance(instance)

                for field in field_members:
                    self.add_field(field, class_definition[field], class_details)

                if field_with_prefix:
                    class_details["_prefix_"] = field_prefix_value

                self.add_details(class_details)
                self.add_dynamic(class_dynamic)

            for class_member in class_members:
                # Catch error
                if class_member == "help":
                    print(
                        f"Invalid syntax: {class_name} must use __doc__ "
                        f"rather than help"
                    )
                    sys.exit(1)
                self.add_class(class_member, class_definition[class_member])

            for class_item in class_items:
                self.add_class(class_item["__class__"], class_item)

        except Exception:
            # traceback.print_exc()
            print(f"Error when processing class {class_name}")

    def add_details(self, class_details):
        if class_details:
            details_lines = json.dumps(class_details, indent=2).splitlines()
            line_start = "self._details_ = "
            for line in details_lines:
                line_with_indent = f"{self.str_indent * 2}{line_start}{line}"
                self.add_line(json_to_python(line_with_indent))
                line_start = ""

    def add_dynamic(self, dynamic):
        if dynamic:
            dynamic_lines = json.dumps(dynamic, indent=2).splitlines()
            line_start = "self._dynamic_ = "
            for line in dynamic_lines:
                line_with_indent = f"{self.str_indent * 2}{line_start}{line}"
                self.add_line(json_to_python(line_with_indent))
                line_start = ""

            self.add_line(f"{self.str_indent * 2}self._process_dynamic()")

    def add_field(self, field_name, field_definition, class_details):
        self.validation_summary.add_field(field_name)
        field_val = None
        if "default" in field_definition:
            field_val = yaml_value(field_definition["default"])
            field_definition["default"] = field_val

        self.add_line(f"{self.str_indent * 2}self.{field_name} = " f"{repr(field_val)}")
        class_details[field_name] = field_definition

    def add_class_instance(self, field_name, instance_definition=None):
        name = self.validation_summary.get_class_name(field_name, instance_definition)
        self.add_line(f"{self.str_indent * 2}self.{field_name} = {name}(self)")

    def add_comment(self, doc_content, str_indent):
        self.add_line(f"{str_indent}'''")
        for line in doc_content.splitlines():
            self.add_line(f"{str_indent}{line}")
        self.add_line(f"{str_indent}'''")

    def add_dict(self, name, d):
        # Adds a dict at file scope
        self.add_separator()
        json_data = json.dumps(d, indent=2)
        line_start = f"{name} = "
        for line in json_data.splitlines():
            self.add_line(json_to_python(f"{line_start}{line}"))
            line_start = ""

    def get_content(self, line_separator="\n"):
        self.content[self.SUMMARY_INDEX] = self.validation_summary.get_summary(
            line_separator
        )
        # Ensure new line at the end
        if self.content[-1]:
            self.content.append("")

        return line_separator.join(self.content)

    def write(self, file_path, line_separator="\n"):
        content = self.get_content(line_separator)
        Path(file_path).write_text(content)


# -----------------------------------------------------------------------------
# API to generate library module
# -----------------------------------------------------------------------------


def generate_module_from_definitions(definitions):
    generated_module = PythonModule()

    for yaml_file in definitions:
        with open(yaml_file) as file:
            yaml_dict = yaml.safe_load(file)

        for key, val in yaml_dict.items():
            generated_module.add_class(key, val)

    return generated_module


# -----------------------------------------------------------------------------
# API to generate CLM key translation dictionary
# -----------------------------------------------------------------------------


def find_paths_to_key(d, search_key):
    """Recursively search a dict for a particular key
    This returns a path to each key found
    """

    def _recursive_find(cur, cur_path, paths):
        if isinstance(cur, dict):
            for key, val in cur.items():
                if key == search_key:
                    paths.append(cur_path)
                    continue
                _recursive_find(val, cur_path + [key], paths)

    result = []
    _recursive_find(d, [], result)
    return result


# -----------------------------------------------------------------------------


def recursive_get(d, path):
    """Get a value in a dictionary from a path"""
    for entry in path:
        d = d[entry]

    return d


# -----------------------------------------------------------------------------


def generate_clm_key_dict(source_file):
    with open(source_file, "r") as rf:
        data = yaml.safe_load(rf)

    paths = find_paths_to_key(data, "clm_key")

    return {recursive_get(data, x + ["clm_key"]): x for x in paths}


# -----------------------------------------------------------------------------
# CLI Main execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    core_definitions = YAML_MODULES_TO_PROCESS
    def_path = Path(__file__).resolve().parent.parent / "definitions"
    definition_files = [def_path / f"{module}.yaml" for module in core_definitions]
    output_file_path = Path(sys.argv[1]).resolve()
    clm_key_file_name = Path(def_path) / "solver.yaml"

    print("-" * 80)
    print("Generate Parflow database module")
    print("-" * 80)
    generated_module = generate_module_from_definitions(definition_files)
    print(generated_module.validation_summary.get_summary())
    # Write out the clm dict as well
    clm_key_dict = generate_clm_key_dict(clm_key_file_name)
    generated_module.add_dict("CLM_KEY_DICT", clm_key_dict)
    print("-" * 80)
    generated_module.write(output_file_path)

    if generated_module.validation_summary.has_duplicate:
        sys.exit(1)
