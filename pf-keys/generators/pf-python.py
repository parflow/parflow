r'''
This module provides the infrastructure to load and generate the ParFlow
database structure as Python classes so IDE and runtime environment can
be used to query the help and constraints associated to each key.
'''

import os
import sys
import traceback
import json
import yaml
from datetime import datetime

# -----------------------------------------------------------------------------

YAML_MODULES_TO_PROCESS = [
    'core',
    'geom',
    'solver',
    'wells',
    'timing',
    'phase',
    'bconditions',
    'netcdf',
    'run'
]

# -----------------------------------------------------------------------------

def is_field(key, definition):
    if key[0] == '_':
        return False

    value = definition[key]

    if '__doc__' in value:
        return False

    if 'help' in value:
        return True

    if '__field__' in value:
        return True

    return False

# -----------------------------------------------------------------------------

def is_class(key, definition):
    if key[0] == '_':
        return False

    if key[0] == '.':
        return False

    value = definition[key]

    if '__doc__' in value:
        return True

    if 'help' in value:
        return False

    if '__field__' in value:
        return False

    return True

# -----------------------------------------------------------------------------

def has_value(key, definition):
    if key[0] == '_':
        return False

    return '__value__' in definition

# -----------------------------------------------------------------------------

def has_prefix(key, definition):
    if key[0] == '_':
        return False

    return '__prefix__' in definition

# -----------------------------------------------------------------------------

def is_class_item(key, definition):
    if key[0] == '.':
        return True

    return False

# -----------------------------------------------------------------------------

def is_dynamic(key, definition):
    item = definition[key]
    return '__class__' in item and '__from__' in item

# -----------------------------------------------------------------------------

def json_to_python(txt):
    return txt.replace(' true,', ' True,').replace(' false,', ' False,').replace(' null', ' None').replace(': true', ': True')

# -----------------------------------------------------------------------------

def yaml_value(yval):
    if isinstance(yval, str):
        try:
            return float(yval)
        except ValueError:
            return yval

    return yval

# -----------------------------------------------------------------------------

class ValidationSummary:
    '''
    This is the class to generate the Python library of ParFlow keys from the
    yaml files.
    '''
    def __init__(self):
        self.has_duplicate = False
        self.class_name_count = {}
        self.class_count = 0
        self.field_count = 0

    def add_class(self, class_name):
        self.class_count += 1
        if class_name in self.class_name_count:
            self.class_name_count[class_name] += 1
        else:
            self.class_name_count[class_name] = 1

        return self.class_name_count[class_name] - 1

    def get_deduplicate_class_name(self, class_name, class_definition=None):
        if class_definition and '__class__' in class_definition:
            return class_definition['__class__']
        if class_name in self.class_name_count:
            return f'{class_name}_{self.class_name_count[class_name]}'
        return class_name

    def get_class_name(self, class_name, class_definition=None):
        if class_definition and '__class__' in class_definition:
            return class_definition['__class__']
        return class_name

    def add_field(self, field_name):
        self.field_count += 1

    def get_summary(self, line_separator='\n'):
        content = [
            f'Created {self.class_count} classes',
        ]
        if len(self.class_name_count) == self.class_count:
            self.has_duplicate = False
            content.append(' => No class name duplication found')
        else:
            self.has_duplicate = True
            content.append(
                f' => We found overlapping class_names ({self.class_count - len(self.class_name_count)})')
            for name in self.class_name_count:
                if self.class_name_count[name] > 1:
                    content.append(
                        f'   + {name} was defined {self.class_name_count[name]} times')

        content.append(f'Defined {self.field_count} fields were found')
        return line_separator.join(content)

    def print_summary(self):
        print(self.get_summary())

# -----------------------------------------------------------------------------

class PythonModule:
    '''
    This class generates the Python library of ParFlow keys from the
    yaml files.
    '''
    def __init__(self, indent=4):
        self.validationSummary = ValidationSummary()
        self.content = [
            "r'''",
            "--- DO NOT EDIT ---",
            "File automatically generated - any manual change will be lost",
            f"Generated on {datetime.now().strftime('%Y/%m/%d - %H:%M:%S')}",
            "",
            "'''",
            "from .core import PFDBObj, PFDBObjListNumber",
        ]
        self.str_indent = ' '*indent

    def add_line(self, content=''):
        self.content.append(content)

    def add_separator(self):
        self.add_line()
        self.add_line(f"# {'-'*78}")
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

            dedup_class_name = self.validationSummary.get_deduplicate_class_name(
                class_name, class_definition)
            self.validationSummary.add_class(self.validationSummary.get_class_name(
                class_name, class_definition))

            inheritance = 'PFDBObj'

            if '__inheritance__' in class_definition:
                inheritance = class_definition['__inheritance__']

            self.add_line(f'class {dedup_class_name}({inheritance}):')
            if '__doc__' in class_keys:
                self.add_comment(class_definition['__doc__'], self.str_indent)

            for key in class_definition:
                if is_class(key, class_definition):
                    class_members.append(key)
                if is_field(key, class_definition):
                    field_members.append(key)
                if key == '__class_instances__':
                    class_instances = class_definition['__class_instances__']
                if is_class_item(key, class_definition):
                    class_items.append(class_definition[key])
                    if '__prefix__' in class_definition[key]:
                        field_with_prefix += 1
                        field_prefix_value = class_definition[key]['__prefix__']
                if is_dynamic(key, class_definition):
                    class_dynamic[class_definition[key]['__class__']] = class_definition[key]['__from__']

            if len(class_members) + len(field_members) + len(class_instances) + field_with_prefix > 0 \
                or len(class_dynamic) \
                or has_prefix(class_name, class_definition):
                '''
                  def __init__(self, parent=None):
                    super().__init__(parent)
                    self.Topology = Topology(self)
                '''
                self.add_line(
                    f'{self.str_indent}def __init__(self, parent=None):')
                self.add_line(f'{self.str_indent*2}super().__init__(parent)')

                if has_value(class_name, class_definition):
                    self.add_field(
                        '_value_', class_definition['__value__'], class_details)

                if has_prefix(class_name, class_definition):
                    self.add_line(f"{self.str_indent * 2}self._prefix_ = '{class_definition['__prefix__']}'")
                    if inheritance == 'PFDBObjListNumber':
                        self.add_line(f"{self.str_indent * 2}self._details_ = ""{}")

                for instance in class_members:
                    self.add_line(
                        f'{self.str_indent*2}self.{instance} = {self.validationSummary.get_deduplicate_class_name(instance, class_definition[instance])}(self)')

                for instance in class_instances:
                    self.add_class_instance(instance) # class_definition[instance]

                for field in field_members:
                    self.add_field(field, class_definition[field], class_details)

                if field_with_prefix:
                    class_details['_prefix_'] = field_prefix_value

                self.add_details(class_details)
                self.add_dynamic(class_dynamic)

            for classMember in class_members:
                # Catch error
                if classMember == 'help':
                    print(
                        f'Invalid syntax: {class_name} must use __doc__ rather than help')
                    sys.exit(1)
                self.add_class(classMember, class_definition[classMember])

            for classItem in class_items:
                self.add_class(classItem['__class__'], classItem)

        except Exception:
            # traceback.print_exc()
            print(f'Error when processing class {class_name}')

    def add_details(self, class_details):
        if len(class_details):
            detailsLines = json.dumps(class_details, indent=2).splitlines()
            self.add_line(
                f'{self.str_indent * 2}self._details_ = {detailsLines[0]}')
            for line in detailsLines[1:]:
                line_with_indent = f'{self.str_indent * 2}{line}'
                self.add_line(json_to_python(line_with_indent))

    def add_dynamic(self, dynamic):
        if len(dynamic):
            dynamicLines = json.dumps(dynamic, indent=2).splitlines()
            self.add_line(
                f'{self.str_indent * 2}self._dynamic_ = {dynamicLines[0]}')
            for line in dynamicLines[1:]:
                line_with_indent = f'{self.str_indent * 2}{line}'
                self.add_line(json_to_python(line_with_indent))
            self.add_line(f'{self.str_indent * 2}self.process_dynamic()')

    def add_field(self, field_name, field_definition, class_details):
        self.validationSummary.add_field(field_name)
        field_val = None
        if 'default' in field_definition:
            field_val = yaml_value(field_definition['default'])
            field_definition['default'] = field_val

        if isinstance(field_val, str):
            self.add_line(f"{self.str_indent*2}self.{field_name} = '{field_val}'")
        else:
            self.add_line(f"{self.str_indent * 2}self.{field_name} = {field_val}")
        class_details[field_name] = field_definition

    def add_class_instance(self, field_name, instance_definition=None):
        self.add_line(
            f"{self.str_indent*2}self.{field_name} = {self.validationSummary.get_class_name(field_name, instance_definition)}(self)")

    def add_comment(self, docContent, str_indent):
        self.add_line(f"{str_indent}'''")
        for line in docContent.splitlines():
            self.add_line(f'{str_indent}{line}')
        self.add_line(f"{str_indent}'''")

    def get_content(self,  line_separator='\n'):
        self.content[4] = self.validationSummary.get_summary(line_separator)
        # Ensure new line at the end
        if len(self.content[-1]):
            self.content.append('')

        return line_separator.join(self.content)

    def write(self, file_path, line_separator='\n'):
        with open(file_path, 'w') as output:
            output.write(self.get_content(line_separator))

# -----------------------------------------------------------------------------
# API to generate library module
# -----------------------------------------------------------------------------

def generate_module_from_definitions(definitions):
    generated_module = PythonModule()

    for yaml_file in definitions:
        with open(yaml_file) as file:
            yaml_dict = yaml.safe_load(file)

            for root_key in yaml_dict.keys():
                generated_module.add_class(root_key, yaml_dict[root_key])

    return generated_module

# -----------------------------------------------------------------------------
# CLI Main execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    core_definitions = YAML_MODULES_TO_PROCESS
    base_path = os.path.dirname(os.path.abspath(__file__))
    def_path = os.path.join(base_path, '../definitions')
    definition_files = [os.path.join(
        def_path, f'{module}.yaml') for module in core_definitions]
    output_file_path = sys.argv[1]

    print('-'*80)
    print('Generate Parflow database module')
    print('-'*80)
    generated_module = generate_module_from_definitions(definition_files)
    print(generated_module.validationSummary.get_summary())
    print('-'*80)
    generated_module.write(output_file_path)

    if generated_module.validationSummary.has_duplicate:
        sys.exit(1)
