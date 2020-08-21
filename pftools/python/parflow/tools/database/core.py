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
    return pfdbObj._parent_

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
    if 'history' in container_obj._details_[name] and len(container_obj._details_[name]['history']):
        history = container_obj._details_[name]['history']
    if 'default' in container_obj._details_[name] and obj == container_obj._details_[name]['default'] and \
            'MandatoryValue' not in container_obj._details_[name]['domains']:
        pass
    else:
        nbErrors, validation_string = validate_value_to_string(name, obj, container_obj._details_[name]['domains'],
                                                               container_obj.get_context_settings(), history, indent)

    return nbErrors, validation_string


# -----------------------------------------------------------------------------

def detail_helper(container, name, value):
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

def is_private_key(name):
    return name[0] == '_' and name[-1] == '_'

# -----------------------------------------------------------------------------

def is_not_private_key(name):
    return not is_private_key(name)

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
        self._parent_ = parent
        self._prefix_ = None

    # ---------------------------------------------------------------------------

    def __setattr__(self, name, value):
        '''
        Helper method that aims to streamline dot notation assignment
        '''
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
                if PFDBObj.exit_on_error:
                    raise ValueError(
                        f'Field "{name}" is not part of the expected schema {self.__class__}')

        # Run domain validation
        if PFDBObj.print_line_error:
            validate_value_with_exception(
                value, domains, PFDBObj.exit_on_error)

        if value_object_assignment:
            self.__dict__[name].__dict__['_value_'] = \
                decorate_value(value, self, handlers)
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
        '''
        Used for obj[] lookup:
           - Need to handle key with prefix
           - Need to handle key with missing prefix
           - Need to handle int key
        '''
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
        '''
        Dynamic help function for runtime evaluation
        '''
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

    def get_key_names(self, skip_default=False):
        '''
        Gets the key names necessary for the run while skiping unset ones
        '''
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
                    if hasattr(obj, '_value_'):
                        value = obj._value_
                        add_errors, validation_string = validate_helper(
                            obj, '_value_', value, indent, error_count)
                        print(f'{indent_str}{name}: {validation_string}')
                        error_count += add_errors
                    else:
                        print(f'{indent_str}{name}:')

                    error_count += obj.validate(indent + 1)

            elif hasattr(self, '_details_') and name in self._details_:
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
        '''
        Helper method returning the key to use for Parflow on a given field key.
        This allow to handle differences between what can be defined in Python vs Parflow key.
        '''
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
        results = []
        for (key, value) in self.__dict__.items():
            if is_private_key(key):
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
          run.Process.Topology.getObjFromLocation('/Geom') => run.Geom
        '''
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
        '''
        Return global settings for our current parflow run.
        This is useful when providing global information for domains or else.
        '''
        return {
            'print_line_error': PFDBObj.print_line_error,
            'exit_on_error': PFDBObj.exit_on_error,
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
            if '_pfstore_' not in self.__dict__:
                self.__dict__['_pfstore_'] = {}
            self.__dict__['_pfstore_'][key] = value

    # ---------------------------------------------------------------------------

    def process_dynamic(self):
        '''
        Processing the dynamically defined (user-defined) key names
        '''
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

    def __setattr__(self, name, value):
        '''
        Helper method that aims to streamline dot notation assignment
        '''
        if is_private_key(name):
            self.__dict__[name] = value
            return

        if self._prefix_:
            if name.startswith(self._prefix_):
                self.__dict__[name] = value
            else:
                self.__dict__[f'{self._prefix_}{name}'] = value
            return

        self.__dict__[name] = value
