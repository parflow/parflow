r'''
This module aims to provide the core components that are required to build
a Parflow input deck.
'''
import os
import re
from .domains import validate_value_with_exception, validate_value_to_string
from .handlers import decorate_value


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def map_to_parent(pfdbObj):
    return pfdbObj._parent

# -----------------------------------------------------------------------------


def map_to_self(pfdbObj):
    return pfdbObj

# -----------------------------------------------------------------------------


def map_to_child(name):
    return lambda pfdbObj: getattr(pfdbObj, name)

# -----------------------------------------------------------------------------


def map_to_children_of_type(class_name):
    def get_children_of_type(pfdbObj):
        return pfdbObj.get_children_of_type(class_name)
    return get_children_of_type

# -----------------------------------------------------------------------------


def validate_helper(container_obj, name, obj, indent, error_count):
    nbErrors = 0
    validation_string = ''
    history = None
    if 'history' in container_obj._details[name] and len(container_obj._details[name]['history']):
        history = container_obj._details[name]['history']
    if 'default' in container_obj._details[name] and obj == container_obj._details[name]['default'] and \
            'MandatoryValue' not in container_obj._details[name]['domains']:
        pass
    else:
        nbErrors, validation_string = validate_value_to_string(name, obj, container_obj._details[name]['domains'],
                                                               container_obj.get_context_settings(), history, indent)

    return nbErrors, validation_string


# -----------------------------------------------------------------------------

def detail_helper(container, name, value):
    domains = None
    handlers = None
    history = None
    crosscheck = None
    if name in container._details:
        if 'domains' in container._details[name]:
            domains = container._details[name]['domains']

        if 'handlers' in container._details[name]:
            handlers = container._details[name]['handlers']

        if 'history' in container._details[name]:
            history = container._details[name]['history']

        else:
            history = []
            container._details[name]['history'] = history
        history.append(value)

        if 'crosscheck' in container._details[name]:
            crosscheck = container._details[name]['crosscheck']

    return domains, handlers, history, crosscheck

# -----------------------------------------------------------------------------
# Main DB Object
# -----------------------------------------------------------------------------


class PFDBObj:
    print_line_error = False
    exit_on_error = False
    working_directory = os.getcwd()
    pf_version = '3.6.0'

    # ---------------------------------------------------------------------------
    # Global settings
    # ---------------------------------------------------------------------------

    @staticmethod
    def enable_line_error():
        PFDBObj.print_line_error = True

    @staticmethod
    def disable_line_error():
        PFDBObj.enable_domain_exceptions = False

    @staticmethod
    def enable_exit_error():
        PFDBObj.exit_on_error = True

    @staticmethod
    def disable_exit_error():
        PFDBObj.exit_on_error = False

    @staticmethod
    def set_working_directory(workdir):
        if workdir:
            PFDBObj.working_directory = workdir
        else:
            PFDBObj.working_directory = os.getcwd()

    @staticmethod
    def set_parflow_version(version):
        PFDBObj.pf_version = version

    # ---------------------------------------------------------------------------
    # Instance specific code
    # ---------------------------------------------------------------------------

    def __init__(self, parent=None):
        '''
        Create container object while keeping a reference to your parent
        '''
        self._parent = parent
        self._prefix = None

    # ---------------------------------------------------------------------------

    def __setattr__(self, name, value):
        '''
        Helper method that aims to streamline dot notation assignment
        '''
        domains = None
        handlers = None
        history = None
        value_object_assignment = False
        if name[0] != '_' and hasattr(self, '_details'):
            if name in self._details:
                domains, handlers, history, crosscheck = detail_helper(
                    self, name, value)
            elif hasattr(self, name) and isinstance(self.__dict__[name], PFDBObj):
                # Handle value object assignment
                value_object_assignment = True
                value_obj = self.__dict__[name]
                domains, handlers, history, crosscheck = detail_helper(
                    value_obj, '_value', value)
            else:
                print(
                    f'Field {name} is not part of the expected schema {self.__class__}')
                if PFDBObj.exit_on_error:
                    raise ValueError(
                        f'Field "{name}" is not part of the expected schema {self.__class__}')

        # Run domain validation
        if PFDBObj.print_line_error:
            validate_value_with_exception(
                value, domains, PFDBObj.exit_on_error)

        if value_object_assignment:
            self.__dict__[name].__dict__[
                '_value'] = decorate_value(value, self, handlers)
        else:
            # Decorate value if need be (i.e. Geom.names: 'a b c')
            self.__dict__[name] = decorate_value(value, self, handlers)

    # ---------------------------------------------------------------------------

    def __len__(self):
        '''
        Return the count of nested fields.
          - If a field is not set but is Mandatory it will count as 1
          - If a field is not set, it will count as 0
          - A container does not count. (0)
        '''
        value_count = 0

        if hasattr(self, '_value') and self._value is not None:
            value_count += 1

        for name in self.get_key_names(True):
            obj = self.__dict__[name]
            if isinstance(obj, PFDBObj):
                value_count += len(obj)
            elif obj is not None:
                value_count += 1
            elif hasattr(self, '_details') and name in self._details and 'domains' in self._details[name]:
                if 'MandatoryValue' in self._details[name]['domains']:
                    value_count += 1

        return value_count

    # ---------------------------------------------------------------------------

    def __getitem__(self, key):
        key_str = str(key)

        if hasattr(self, key_str):
            return getattr(self, key_str)

        prefix = '_'
        if self._details and '_prefix' in self._details:
            prefix = self._details['_prefix']

        key_str = f'{prefix}{key_str}'
        if hasattr(self, key_str):
            return getattr(self, key_str)

        print(f'Could not find key {key_str} in {self.__dict__.keys()}')
        return getattr(self, key_str)

    # ---------------------------------------------------------------------------

    def help(self, key=None):
        '''
        Dynamic help function for runtime evaluation
        '''
        if key is not None:
            if key in self._details:
                if 'help' in self._details[key]:
                    print(self._details[key]['help'])
            else:
                obj = self.__dict__[key]
                if hasattr(obj, '__doc__'):
                    print(obj.__doc__)

                if hasattr(obj, '_details') and '_value' in obj._details and 'help' in obj._details['_value']:
                    print(obj._details['_value']['help'])

        elif hasattr(self, '__doc__'):
            print(self.__doc__)
            if hasattr(self, '_details') and '_value' in self._details and 'help' in self._details['_value']:
                print(self._details['_value']['help'])

    # ---------------------------------------------------------------------------

    def get_key_names(self, skip_default=False):
        for name in self.__dict__:
            if name is None:
                print('need to fix the children instantiator')
                continue

            if name[0] == '_' and name[1].isalpha():
                # if name[1].isdigit():
                #     print(name)
                continue

            obj = self.__dict__[name]
            if isinstance(obj, PFDBObj):
                if len(obj):
                    yield name

            else:
                has_details = hasattr(
                    self, '_details') and name in self._details
                has_default = has_details and 'default' in self._details[name]
                has_domain = has_details and 'domains' in self._details[name]
                is_mandatory = has_domain and 'MandatoryValue' in self._details[name]['domains']
                is_default = has_default and obj == self._details[name]['default']

                if obj is not None:
                    if skip_default:
                        if not is_default or is_mandatory:
                            yield name
                    else:
                        yield name

                elif is_mandatory:
                    yield name

    # ---------------------------------------------------------------------------

    def validate(self, indent=1, workdir=None):
        '''
        Method to validate sub hierarchy
        '''
        if len(self) == 0:
            return 0

        error_count = 0
        indent_str = '  '*indent
        for name in self.get_key_names(skip_default=True):
            obj = self.__dict__[name]
            if isinstance(obj, PFDBObj):
                if len(obj):
                    if hasattr(obj, '_value'):
                        value = obj._value
                        add_errors, validation_string = validate_helper(
                            obj, '_value', value, indent, error_count)
                        print(f'{indent_str}{name}: {validation_string}')
                        error_count += add_errors
                    else:
                        print(f'{indent_str}{name}:')

                    error_count += obj.validate(indent + 1)

            elif hasattr(self, '_details') and name in self._details:
                add_errors, validation_string = validate_helper(
                    self, name, obj, indent, error_count)
                print(f'{indent_str}{name}: {validation_string}')
                error_count += add_errors
            elif obj is not None:
                print(f'{indent_str}{name}: {obj}')

        return error_count

    # ---------------------------------------------------------------------------

    def get_full_key_name(self):
        '''
        Helper method returning the full name of a given ParFlow key.
        '''
        full_path = []
        current_location = self
        count = 0
        while current_location._parent is not None:
            count += 1
            parent = current_location._parent
            for name in parent.__dict__:
                value = parent.__dict__[name]
                if value == current_location:
                    full_path.append(name)
            current_location = parent
            if count > len(full_path):
                return f'not found {count}: {".".join(full_path)}'

        full_path.reverse()
        return '.'.join(full_path)

    # ---------------------------------------------------------------------------

    def get_parflow_key(self, parent_namespace, key):
        '''
        Helper method returning the key to use for Parflow on a given field key.
        This allow to handle differences between what can be defined in Python vs Parflow key.
        '''
        value = self.__dict__[key]
        prefix = ''
        if isinstance(value, PFDBObj):
            if value._prefix and key.startswith(value._prefix):
                prefix = value._prefix
        else:
            print(value)
            detail = self._details[key]
            if '_prefix' in detail:
                prefix = detail["_prefix"]

        if parent_namespace:
            return f'{parent_namespace}.{key[len(prefix):]}'

        return key[len(prefix):]

    # ---------------------------------------------------------------------------

    def get_children_of_type(self, class_name):
        results = []
        for (key, value) in self.__dict__.items():
            if key[0] == '_':
                continue
            if value.__class__.__name__ == class_name:
                results.append(value)

        return results

    # ---------------------------------------------------------------------------

    def get_selection_from_location(self, location='.'):
        '''
        Return a PFDBObj object based on a location.

        i.e.:
          run.Process.Topology.getObjFromLocation('.') => run.Process.Topology
          run.Process.Topology.getObjFromLocation('..') => run.Process
          run.Process.Topology.getObjFromLocation('../../Geom') => run.Geom
        '''
        current_location = self
        path_items = location.split('/')
        if location[0] == '/':
            while current_location._parent is not None:
                current_location = current_location._parent

        next_list = [current_location]
        for path_item in path_items:
            if path_item == '':
                continue

            # print(f'>>> List: {next_list}')
            # print(f'>>> Path: {path_item}')

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

        # print(f'=>{next_list}')
        return next_list

    # ---------------------------------------------------------------------------

    def get_context_settings(self):
        '''
        Return global settings for our current parflow run.
        This is useful when providing global information for domains or else.
        '''
        return {
            'print_line_error': PFDBObj.print_line_error,
            'exitOnError': PFDBObj.exit_on_error,
            'working_directory': PFDBObj.working_directory,
            'pf_version': PFDBObj.pf_version
        }

    # ---------------------------------------------------------------------------

    def pfset(self, key='', value=None):
        '''
        Allow to define any parflow key so it can be exported
        '''
        tokens = key.split('.')
        container = self.get_selection_from_location('/'.join(tokens[:-1]))[0]
        if container:
            container[tokens[-1]] = value
        else:
            # store key on the side
            if '_pfstore' not in self.__dict__:
                self.__dict__['_pfstore'] = {}
            self.__dict__['_pfstore'][key] = value

    # ---------------------------------------------------------------------------

    def process_dynamic(self):
        '''
        Processing the dynamically defined (user-defined) key names
        '''
        from . import generated
        for (class_name, selection) in self._dynamic.items():
            klass = getattr(generated, class_name)
            names = self.get_selection_from_location(selection)
            for name in names:
                if name is not None:
                    self.__dict__[name] = klass(self)

# -----------------------------------------------------------------------------
# Main DB Object
# -----------------------------------------------------------------------------

class PFDBObjListNumber(PFDBObj):

    def __setattr__(self, name, value):
        '''
        Helper method that aims to streamline dot notation assignment
        '''
        if name[0] == '_':
            self.__dict__[name] = value
            return

        if self._prefix:
            if name.startswith(self._prefix):
                self.__dict__[name] = value
            else:
                self.__dict__[f'{self._prefix}{name}'] = value
            return

        self.__dict__[name] = value

    # def get_parflow_key(self, parent_namespace, key):
    #     '''
    #     Helper method returning the key to use for Parflow on a given field key.
    #     This allow to handle differences between what can be defined in Python vs Parflow key.
    #     '''
    #     value = self.__dict__[key]
    #     prefix = ''
    #     if isinstance(value, PFDBObj):
    #         if value._prefix and key.startswith(value._prefix):
    #             prefix = value._prefix
    #     else:
    #         detail = self._details[key]
    #         if '_prefix' in detail:
    #             prefix = detail["_prefix"]
    #
    #     if parent_namespace:
    #         return f'{parent_namespace}.{key[len(prefix):]}'
    #
    #     return key[len(prefix):]