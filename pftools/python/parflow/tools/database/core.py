r"""
This module aims to provide the core components that are required to build
a Parflow input deck.
"""
import json
import os
import re
import sys
import yaml

try:
    from yaml import CDumper as YAMLDumper
except ImportError:
    from yaml import Dumper as YAMLDumper

from parflow.tools import settings
from parflow.tools.fs import get_text_file_content
from parflow.tools.helper import map_to_child, map_to_children_of_type, map_to_parent, map_to_self
from parflow.tools.helper import sort_dict_by_priority

from .domains import validate_value_to_string, validate_value_with_exception
from .handlers import decorate_value

# -----------------------------------------------------------------------------
# Accessor helpers
# -----------------------------------------------------------------------------

def validate_helper(container_obj, name, value, indent):
    """Helper function for validating a value
    """
    nbErrors = 0
    validation_string = ''
    has_default = True if 'default' in container_obj._details_[name] else False
    history = None
    if 'history' in container_obj._details_[name] and len(container_obj._details_[name]['history']):
        history = container_obj._details_[name]['history']
    if 'default' in container_obj._details_[name] \
            and value == container_obj._details_[name]['default'] \
            and 'MandatoryValue' not in container_obj._details_[name]['domains']:
        pass
    else:
        nbErrors, validation_string = validate_value_to_string(container_obj, value, has_default, container_obj._details_[name]['domains'],
                                                               container_obj.get_context_settings(), history, indent)

    return nbErrors, validation_string

# -----------------------------------------------------------------------------


def detail_helper(container, name, value):
    """Helper function that extract elements of the field's detail"""
    domains = None
    handlers = None
    history = None
    crosscheck = None
    if name in container._details_:
        if 'domains' in container._details_[name]:
            domains = container._details_[name]['domains']

        if 'handlers' in container._details_[name]:
            handlers = container._details_[name]['handlers']

        if 'history' in container._details_[name]:
            history = container._details_[name]['history']

        else:
            history = []
            container._details_[name]['history'] = history
        history.append(value)

        if 'crosscheck' in container._details_[name]:
            crosscheck = container._details_[name]['crosscheck']

    return domains, handlers, history, crosscheck

# -----------------------------------------------------------------------------
# Internal field name helpers
# -----------------------------------------------------------------------------

def is_private_key(name):
    """Test if the given key is a key or a private member

    Return True if it is a private member
    """
    return name[0] == '_' and name[-1] == '_'

# -----------------------------------------------------------------------------

def is_not_private_key(name):
    """Test if the given key is a key or a private member

    Return True if it is a key
    """
    return not is_private_key(name)

# -----------------------------------------------------------------------------

def convert_value_for_string_dict(value):
    """Ensure that the output value is a valid string
    """
    if isinstance(value, str):
        return value

    if hasattr(value, '__iter__'):
        return ' '.join([str(v) for v in value])

    return value

# -----------------------------------------------------------------------------

def extract_keys_from_object(dict_to_fill, instance, parent_namespace=''):
    """Method that walk PFDBObj object and record their key and value
    inside a Python dict.
    """
    if hasattr(instance, '_pfstore_'):
        for key, value in instance._pfstore_.items():
            dict_to_fill[key] = convert_value_for_string_dict(value)

    for key in instance.get_key_names(skip_default=True):
        value = instance[key]
        if value is None:
            continue

        full_qualified_key = instance.get_parflow_key(parent_namespace, key)
        if isinstance(value, PFDBObj):
            if hasattr(value, '_value_'):
                has_details = hasattr(value, '_details_') \
                    and '_value_' in value._details_
                has_default = has_details \
                    and 'default' in value._details_['_value_']
                has_domain = has_details \
                    and 'domains' in value._details_['_value_']
                is_mandatory = has_domain \
                    and 'MandatoryValue' in value._details_['_value_']['domains']
                is_default = has_default \
                    and value._value_ == value._details_['_value_']['default']
                is_set = has_details \
                    and 'history' in value._details_['_value_'] \
                    and len(value._details_['_value_']['history']) > 0

                if is_mandatory or not is_default or is_set:
                    dict_to_fill[full_qualified_key] = \
                        convert_value_for_string_dict(value._value_)
            extract_keys_from_object(dict_to_fill, value, full_qualified_key)
        else:
            dict_to_fill[full_qualified_key] = \
                convert_value_for_string_dict(value)

# -----------------------------------------------------------------------------

def extract_keys_from_dict(dict_to_fill, dictObj, parent_namespace=''):
    """Helper function to extract a flat key/value dictionary for
    a given PFDBObj inside dict_to_fill.
    """
    for key, value in dictObj.items():
        if len(parent_namespace) and key == '$_' or key == '_value_':
            dict_to_fill[parent_namespace] = value
            continue

        if value is None or is_private_key(key):
            continue

        full_qualified_key = f'{parent_namespace}.{key}' if parent_namespace else key
        if isinstance(value, dict):
            # Need to handle _value_ and $_
            if hasattr(value, '_value_'):
                dict_to_fill[full_qualified_key] = convert_value_for_string_dict(
                    value._value_)
            if hasattr(value, '$_'):
                dict_to_fill[full_qualified_key] = convert_value_for_string_dict(
                    value._value_)
            extract_keys_from_dict(dict_to_fill, value, full_qualified_key)
        else:
            dict_to_fill[full_qualified_key] = convert_value_for_string_dict(
                value)

# -----------------------------------------------------------------------------

def flatten_hierarchical_map(hirearchical_map):
    """Helper function that take a hierarchical map and return a flat
    version of it.
    """
    flat_map = {}
    extract_keys_from_dict(flat_map, hirearchical_map, parent_namespace='')
    return flat_map

# -----------------------------------------------------------------------------
# Main DB Object
# -----------------------------------------------------------------------------

class PFDBObj:
    """Core ParFlow Database Object node
    """
    def __init__(self, parent=None):
        """
        Create container object while keeping a reference to your parent
        """
        self._parent_ = parent
        self._prefix_ = None

    # ---------------------------------------------------------------------------

    def __setitem__(self, key, value):
        """Allow a[x] for assignment as well"""
        self.__setattr__(key, value)

    # ---------------------------------------------------------------------------

    def __setattr__(self, name, value):
        """
        Helper method that aims to streamline dot notation assignment
        """
        domains = None
        handlers = None
        history = None
        value_object_assignment = False
        if is_not_private_key(name) and hasattr(self, '_details_'):
            if name in self._details_:
                domains, handlers, history, crosscheck = detail_helper(
                    self, name, value)
            elif hasattr(self, name) and isinstance(self.__dict__[name], PFDBObj):
                # Handle value object assignment
                value_object_assignment = True
                value_obj = self.__dict__[name]
                domains, handlers, history, crosscheck = detail_helper(
                    value_obj, '_value_', value)
            else:
                print(
                    f'Field {name} is not part of the expected schema {self.__class__}')
                if settings.EXIT_ON_ERROR:
                    raise ValueError(
                        f'Field "{name}" is not part of the expected schema {self.__class__}')

        # Run domain validation
        if settings.PRINT_LINE_ERROR:
            validate_value_with_exception(
                value, domains, settings.EXIT_ON_ERROR)

        if value_object_assignment:
            self.__dict__[name].__dict__['_value_'] = \
                decorate_value(value, self, handlers)
        else:
            # Decorate value if need be (i.e. Geom.names: 'a b c')
            self.__dict__[name] = decorate_value(value, self, handlers)

    # ---------------------------------------------------------------------------

    def __len__(self):
        """
        Return the count of nested fields.
          - If a field is not set but is Mandatory it will count as 1
          - If a field is not set, it will count as 0
          - A container does not count. (0)
        """
        value_count = 0

        if hasattr(self, '_value_') and self._value_ is not None:
            value_count += 1

        for name in self.get_key_names(True):
            obj = self.__dict__[name]
            if isinstance(obj, PFDBObj):
                value_count += len(obj)
            elif obj is not None:
                value_count += 1
            elif hasattr(self, '_details_') and name in self._details_ \
                    and 'domains' in self._details_[name]:
                if 'MandatoryValue' in self._details_[name]['domains']:
                    value_count += 1

        return value_count

    # ---------------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Used for obj[] lookup:
           - Need to handle key with prefix
           - Need to handle key with missing prefix
           - Need to handle int key
        """
        key_str = str(key)

        if hasattr(self, key_str):
            return getattr(self, key_str)

        prefix = ''
        if self._details_ and '_prefix_' in self._details_:
            prefix = self._details_['_prefix_']

        key_str = f'{prefix}{key_str}'
        if hasattr(self, key_str):
            return getattr(self, key_str)

        print(f'Could not find key {key}/{key_str} in {self.__dict__.keys()}')
        return getattr(self, key_str)

    # ---------------------------------------------------------------------------

    def help(self, key=None):
        """
        Dynamic help function for runtime evaluation
        """
        if key is not None:
            if key in self._details_:
                if 'help' in self._details_[key]:
                    print(self._details_[key]['help'])
            else:
                obj = self.__dict__[key]
                if hasattr(obj, '__doc__'):
                    print(obj.__doc__)

                if hasattr(obj, '_details_') and '_value_' in obj._details_ \
                        and 'help' in obj._details_['_value_']:
                    print(obj._details_['_value_']['help'])

        elif hasattr(self, '__doc__'):
            print(self.__doc__)
            if hasattr(self, '_details_') and '_value_' in self._details_ \
                    and 'help' in self._details_['_value_']:
                print(self._details_['_value_']['help'])

    # ---------------------------------------------------------------------------

    def get_key_dict(self):
        """Method that will return a flat map of all the ParFlow keys.

        Returns:
          dict: Return Python dict with all the key set listed without
              any hierarchy.
        """
        key_dict = {}
        extract_keys_from_object(key_dict, self)
        return key_dict

    # ---------------------------------------------------------------------------

    def get_key_names(self, skip_default=False):
        """
        Gets the key names necessary for the run while skiping unset ones
        """
        for name in self.__dict__:
            if name is None:
                print('need to fix the children instantiator')
                continue

            if is_private_key(name):
                continue

            obj = self.__dict__[name]
            if isinstance(obj, PFDBObj):
                if len(obj):
                    yield name

            else:
                has_details = hasattr(self, '_details_') \
                    and name in self._details_
                has_default = has_details \
                    and 'default' in self._details_[name]
                has_domain = has_details \
                    and 'domains' in self._details_[name]
                is_mandatory = has_domain \
                    and 'MandatoryValue' in self._details_[name]['domains']
                is_default = has_default \
                    and obj == self._details_[name]['default']
                is_set = has_details \
                    and 'history' in self._details_[name] \
                    and len(self._details_[name]['history']) > 0

                if obj is not None:
                    if skip_default:
                        if not is_default or is_mandatory or is_set:
                            yield name
                    else:
                        yield name

                elif is_mandatory:
                    yield name

    # ---------------------------------------------------------------------------

    def validate(self, indent=1, workdir=None):
        """
        Method to validate sub hierarchy
        """
        if len(self) == 0:
            return 0

        error_count = 0
        indent_str = '  '*indent
        for name in self.get_key_names(skip_default=True):
            obj = self.__dict__[name]
            if isinstance(obj, PFDBObj):
                if len(obj):
                    if hasattr(obj, '_value_'):
                        value = obj._value_
                        add_errors, validation_string = validate_helper(
                            obj, '_value_', value, indent)
                        print(f'{indent_str}{name}: {validation_string}')
                        error_count += add_errors
                    else:
                        print(f'{indent_str}{name}:')

                    error_count += obj.validate(indent + 1)

            elif hasattr(self, '_details_') and name in self._details_:
                add_errors, validation_string = validate_helper(
                    self, name, obj, indent)
                print(f'{indent_str}{name}: {validation_string}')
                error_count += add_errors
            elif obj is not None:
                print(f'{indent_str}{name}: {obj}')

        return error_count

    # ---------------------------------------------------------------------------

    def get_full_key_name(self):
        """
        Helper method returning the full name of a given ParFlow key.
        """
        full_path = []
        current_location = self
        count = 0
        while current_location._parent_ is not None:
            count += 1
            parent = current_location._parent_
            for name in parent.__dict__:
                value = parent.__dict__[name]
                if value == current_location:
                    if current_location._prefix_:
                        full_path.append(name[len(current_location._prefix_):])
                    else:
                        full_path.append(name)
            current_location = parent
            if count > len(full_path):
                return f'not found {count}: {".".join(full_path)}'

        full_path.reverse()
        return '.'.join(full_path)

    # ---------------------------------------------------------------------------

    def get_parflow_key(self, parent_namespace, key):
        """
        Helper method returning the key to use for Parflow on a given field key.
        This allow to handle differences between what can be defined in Python vs Parflow key.
        """
        value = self.__dict__[key]
        prefix = ''
        if isinstance(value, PFDBObj):
            if value._prefix_ and key.startswith(value._prefix_):
                prefix = value._prefix_
        else:
            if key in self._details_:
                detail = self._details_[key]
                if '_prefix_' in detail:
                    prefix = detail["_prefix_"]

        if parent_namespace:
            return f'{parent_namespace}.{key[len(prefix):]}'

        return key[len(prefix):]

    # ---------------------------------------------------------------------------

    def get_children_of_type(self, class_name):
        """Return a list of PFDBObj of a given type that are part of
        our children.
        """
        results = []
        for (key, value) in self.__dict__.items():
            if is_private_key(key):
                continue
            if value.__class__.__name__ == class_name:
                results.append(value)

        return results

    # ---------------------------------------------------------------------------

    def get_selection_from_location(self, location='.'):
        """
        Return a PFDBObj object based on a location.

        i.e.:
          run.Process.Topology.get_selection_from_location('.') => run.Process.Topology
          run.Process.Topology.get_selection_from_location('..') => run.Process
          run.Process.Topology.get_selection_from_location('../../Geom') => run.Geom
          run.Process.Topology.get_selection_from_location('/Geom') => run.Geom
        """
        current_location = self
        path_items = location.split('/')
        if location[0] == '/':
            while current_location._parent_ is not None:
                current_location = current_location._parent_

        next_list = [current_location]
        for path_item in path_items:
            if path_item == '':
                continue

            current_list = next_list
            next_list = []

            if path_item == '..':
                next_list.extend(map(map_to_parent, current_list))
            elif path_item == '.':
                next_list.extend(map(map_to_self, current_list))
            elif path_item[0] == '{':
                multiList = map(map_to_children_of_type(
                    path_item[1:-1]), current_list)
                next_list = [item for sublist in multiList for item in sublist]
            else:
                next_list.extend(map(map_to_child(path_item), current_list))
                if len(next_list) and isinstance(next_list[0], list):
                    next_list = [
                        item for sublist in next_list for item in sublist]

        return next_list

    # ---------------------------------------------------------------------------

    def get_context_settings(self):
        """
        Return global settings for our current parflow run.
        This is useful when providing global information for domains or else.
        """
        return {
            'print_line_error': settings.PRINT_LINE_ERROR,
            'exit_on_error': settings.EXIT_ON_ERROR,
            'working_directory': settings.WORKING_DIRECTORY,
            'pf_version': settings.PARFLOW_VERSION
        }

    # ---------------------------------------------------------------------------

    def pfset(self, key='', value=None, yamlFile=None, yamlContent=None, hierarchical_map=None, flat_map=None, exit_if_undefined=False):
        """
        Allow to define any parflow key so it can be exported. Many format are supported:
            - key/value: To set a single value relative to our current
                PFDBObj.
            - yamlFile: YAML file path to load and import using the
                current PFDBObj as root.
            - yamlContent: YAML string to load and import using the
                current PFDBObj as root.
            - hierarchical_map: Nested dict containing several key/value
                pair using the current PFDBObj as root.
            - flat_map: Flat dict with parflow key/value pair to set
                using the current PFDBObj as root.
        """
        if yamlFile:
            yamlContent = get_text_file_content(yamlFile)

        if yamlContent:
            hierarchical_map = yaml.safe_load(yamlContent)

        if hierarchical_map:
            flat_map = flatten_hierarchical_map(hierarchical_map)

        if flat_map:
            sorted_flat_map = sort_dict_by_priority(flat_map)
            for key, value in sorted_flat_map.items():
                self.pfset(key=key, value=value,
                           exit_if_undefined=exit_if_undefined)

        if not key:
            return

        # print(f'{key} = {value}')

        key_stored = False
        tokens = key.split('.')
        if len(tokens) > 1:
            container = self.get_selection_from_location(
                '/'.join(tokens[:-1]))[0]
            if container is not None:
                container[tokens[-1]] = value
                key_stored = True
        elif len(tokens) == 1:
            self[tokens[0]] = value
            key_stored = True

        if not key_stored:
            # store key on the side
            if '_pfstore_' not in self.__dict__:
                self.__dict__['_pfstore_'] = {}
            parentNamespace = self.get_full_key_name()
            fullkeyName = f"{parentNamespace}{'.' if parentNamespace else ''}{key}"
            self.__dict__['_pfstore_'][fullkeyName] = value
            rootPath = self.get_full_key_name()
            print(f"Caution: Using internal store of {rootPath if rootPath else 'run'} to save {fullkeyName} = {value}")
            if exit_if_undefined:
                sys.exit(1)

    # ---------------------------------------------------------------------------

    def process_dynamic(self):
        """
        Processing the dynamically defined (user-defined) key names
        """
        from . import generated
        for (class_name, selection) in self._dynamic_.items():
            klass = getattr(generated, class_name)
            names = self.get_selection_from_location(selection)
            for name in names:
                if name is not None:
                    self.__dict__[name] = klass(self)

# -----------------------------------------------------------------------------
# Main DB Object
# -----------------------------------------------------------------------------

class PFDBObjListNumber(PFDBObj):
    """Class for leaf list values"""

    def __setattr__(self, name, value):
        """
        Helper method that aims to streamline dot notation assignment
        """
        key_str = str(name)
        if is_private_key(key_str):
            self.__dict__[key_str] = value
            return

        if self._prefix_:
            if key_str.startswith(self._prefix_):
                self.__dict__[key_str] = value
            else:
                self.__dict__[f'{self._prefix_}{key_str}'] = value
            return

        self.__dict__[key_str] = value

    def get_parflow_key(self, parent_namespace, key):
        """
        Helper method returning the key to use for Parflow on a given field key.
        This allow to handle differences between what can be defined in Python vs Parflow key.
        """
        prefix = self._prefix_ if self._prefix_ else ''

        if parent_namespace:
            return f'{parent_namespace}.{key[len(prefix):]}'

        return key[len(prefix):]
