r'''
This module provide the infrastructure to load and generate the Parflow
database structure as documentation files for Read The Docs.
'''

import os
import yaml

# -----------------------------------------------------------------------------

YAML_MODULES_TO_PROCESS = [
    'core',
    'geom',
    'solver',
    'wells',
    'phase',
    'timing',
    'netcdf',
    'bconditions',
    'run'
]

# -----------------------------------------------------------------------------

LEVELS = [
    '=',
    '-',
    '^',
    '"',
    '"',
    '"',
    '"',
]

# -----------------------------------------------------------------------------

def handle_domain(name, definition):
    '''
    This method will extract information from a domain and present it
    for the documentation.
    '''
    indent_str = ' '*4
    lines = []
    list_count = 0

    if name == 'MandatoryValue':
        lines.append(f'{indent_str}The value is required')

    if name == 'IntValue':
        lines.append(f'{indent_str}The value must be an Integer')
        if definition and 'min_value' in definition:
            list_count += 1
            lines.append(
                f'{indent_str}  - with a value greater than or equal to {definition["min_value"]}')
        if definition and 'max_value' in definition:
            list_count += 1
            lines.append(
                f'{indent_str}  - with a value less than or equal to {definition["max_value"]}')

    if name == 'DoubleValue':
        lines.append(f'{indent_str}The value must be a Double')
        if definition and 'min_value' in definition:
            list_count += 1
            lines.append(
                f'{indent_str}  - with a value greater than or equal to {definition["min_value"]}')
        if definition and 'max_value' in definition:
            list_count += 1
            lines.append(
                f'{indent_str}  - with a value less than or equal to {definition["max_value"]}')
        if definition and 'neg_int' in definition:
            list_count += 1
            lines.append(
                f'{indent_str}  - must be an integer if less than 0')

    if name == 'EnumDomain':
        lines.append(
            f'{indent_str}The value must be one of the following options: {(", ".join(definition["enum_list"]))}')

    if name == 'AnyString':
        lines.append(f'{indent_str}The value must be a string')

    if name == 'BoolDomain':
        lines.append(f'{indent_str}The value must be True or False')

    if name == 'RequiresModule':
        lines.append(
            f'{indent_str}This key requires the availability of the following module(s) in ParFlow: {definition}')

    if name == 'Deprecated':
        lines.append('')
        lines.append('.. warning::')
        lines.append(f'    This key will be deprecated in v{definition}')

    if name == 'Removed':
        lines.append('')
        lines.append('.. warning::')
        lines.append(f'    This key will be removed in v{definition}')

    if list_count:
        lines.append('')

    return '\n'.join(lines)

# -----------------------------------------------------------------------------

class RST_module:
    '''
    Helper class that can be used to create a RST file for ReadTheDoc
    '''
    def __init__(self, title):
        self.content = [
            '*'*80,
            title,
            '*'*80,
        ]

    def add_line(self, content=''):
        self.content.append(content)

    def add_section(self, level, prefix, key, sub_section):
        if prefix and prefix != 'BaseRun':
            title = f'{prefix}.{key}'
        else:
            title = key

        if key == '__value__':
            title = prefix

        warning = ''
        if '__rst__' in sub_section:
            if 'name' in sub_section['__rst__']:
                title = sub_section['__rst__']['name']
            if 'warning' in sub_section['__rst__']:
                warning = sub_section['__rst__']['warning']
            if 'skip' in sub_section['__rst__']:
                for sub_key in sub_section:
                    if sub_key[0] != '_' or sub_key == '__value__':
                        self.add_section(level, title, sub_key,
                                         sub_section[sub_key])
                return

        self.add_line()
        self.add_line(title)
        self.add_line(LEVELS[level]*80)
        self.add_line()
        if warning:
            self.add_line('.. warning::')
            self.add_line(f'    {warning}')

        leaf = False
        description = ''

        if 'help' in sub_section:
            leaf = True
            description = sub_section['help']

        if '__doc__' in sub_section:
            description = sub_section['__doc__']

        self.add_line(description)
        self.add_line()

        if leaf:
            # Need to process domains and more...
            if 'default' in sub_section:
                self.add_line(f':default: {sub_section["default"]}')

            if 'domains' in sub_section:
                self.add_line('.. note::')
                for domain in sub_section['domains']:
                    self.add_line(handle_domain(
                        domain, sub_section['domains'][domain]))
                self.add_line()

        else:
            # Keep adding sections
            for sub_key in sub_section:
                if sub_key[0] != '_' or sub_key == '__value__':
                    self.add_section(level + 1, title, sub_key,
                                     sub_section[sub_key])

    def get_content(self,  line_separator='\n'):
        # Ensure new line at the end
        if len(self.content[-1]):
            self.content.append('')

        return line_separator.join(self.content)

    def write(self, file_path, line_separator='\n'):
        with open(file_path, 'w') as output:
            output.write(self.get_content(line_separator))

# -----------------------------------------------------------------------------
# Expected API to use
# -----------------------------------------------------------------------------

def generate_module_from_definitions(definitions):
    generated_RST = RST_module('ParFlow Key Documentation')

    for yaml_file in definitions:
        with open(yaml_file) as file:
            yaml_struct = yaml.safe_load(file)

            for root_key in yaml_struct.keys():
                generated_RST.add_section(0, '', root_key, yaml_struct[root_key])

    return generated_RST

# -----------------------------------------------------------------------------
# CLI Main execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    core_definitions = YAML_MODULES_TO_PROCESS
    base_path = os.path.dirname(os.path.abspath(__file__))
    print(base_path)
    defPath = os.path.join(base_path, '../definitions')
    definition_files = [os.path.join(
        defPath, f'{module}.yaml') for module in core_definitions]
    output_file_path = os.path.join(base_path, '../../docs/pf-keys/parflow_keys.rst')

    print('-'*80)
    print('Generate ParFlow database documentation')
    print('-'*80)
    generated_module = generate_module_from_definitions(definition_files)
    print('-'*80)
    generated_module.write(output_file_path)
