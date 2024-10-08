r"""
This module aims to provide the core components that are required to build
a Parflow input deck.
"""

import sys
import yaml

from parflow.tools import settings
from parflow.tools.fs import get_text_file_content
from parflow.tools.helper import (
    map_to_child,
    map_to_children_of_type,
    map_to_parent,
    map_to_self,
    remove_prefix,
    filter_none,
)
from parflow.tools.helper import normalize_location, sort_dict_by_priority
from parflow.tools.io import read_pfidb

from .domains import validate_value_to_string, validate_value_with_exception
from .handlers import decorate_value


# -----------------------------------------------------------------------------
# Accessor helpers
# ----------------------------------------------------------------------------


def validate_helper(container, name, value, indent):
    """Helper function for validating a value"""
    num_errors = 0
    validation_string = ""
    details = container._details_[name]
    has_default = "default" in details
    history = details.get("history")
    if (
        has_default
        and value == details["default"]
        and "MandatoryValue" not in details["domains"]
    ):
        pass
    else:
        num_errors, validation_string = validate_value_to_string(
            container,
            value,
            has_default,
            details["domains"],
            container.get_context_settings(),
            history,
            indent,
        )

    return num_errors, validation_string


# -----------------------------------------------------------------------------


def detail_helper(container, name, value):
    """Helper function that extract elements of the field's detail"""
    details = container._details_.get(name, {})
    domains = details.get("domains")
    handlers = details.get("handlers")
    crosscheck = details.get("crosscheck")
    if details:
        history = details.setdefault("history", [])
        history.append(value)
    else:
        history = None

    return domains, handlers, history, crosscheck


# -----------------------------------------------------------------------------
# Internal field name helpers
# -----------------------------------------------------------------------------


def is_private_key(name):
    """Test if the given key is a key or a private member

    Return True if it is a private member
    """
    return name[0] == "_" and name[-1] == "_"


# -----------------------------------------------------------------------------


def is_not_private_key(name):
    """Test if the given key is a key or a private member

    Return True if it is a key
    """
    return not is_private_key(name)


# -----------------------------------------------------------------------------


def to_str_dict_format(value):
    """Ensure that the output value is a valid string"""
    if isinstance(value, str):
        return value

    if hasattr(value, "__iter__"):
        return " ".join([str(v) for v in value])

    return value


# -----------------------------------------------------------------------------


def extract_keys_from_object(dict_to_fill, instance, parent_namespace=""):
    """Method that walk PFDBObj object and record their key and value
    inside a Python dict.
    """
    if hasattr(instance, "_pfstore_"):
        for key, value in instance._pfstore_.items():
            dict_to_fill[key] = to_str_dict_format(value)

    for key in instance.keys(skip_default=True):
        value = instance[key]
        if value is None:
            continue

        full_qualified_key = instance.to_pf_name(parent_namespace, key)
        if isinstance(value, PFDBObj):
            if hasattr(value, "_value_"):
                has_details = (
                    hasattr(value, "_details_") and "_value_" in value._details_
                )
                details = value._details_["_value_"] if has_details else None
                has_default = has_details and "default" in details
                has_domain = has_details and "domains" in details
                is_mandatory = has_domain and "MandatoryValue" in details["domains"]
                is_default = has_default and value._value_ == details["default"]
                is_set = has_details and details.get("history")
                if is_mandatory or not is_default or is_set:
                    dict_to_fill[full_qualified_key] = to_str_dict_format(value._value_)
            extract_keys_from_object(dict_to_fill, value, full_qualified_key)
        else:
            dict_to_fill[full_qualified_key] = to_str_dict_format(value)


# -----------------------------------------------------------------------------


def extract_keys_from_dict(dict_to_fill, dict_obj, parent_namespace=""):
    """Helper function to extract a flat key/value dictionary for
    a given PFDBObj inside dict_to_fill.
    """
    for key, value in dict_obj.items():
        if parent_namespace and key == "_value_":
            dict_to_fill[parent_namespace] = value
            continue

        if value is None or is_private_key(key):
            continue

        full_qualified_key = f"{parent_namespace}.{key}" if parent_namespace else key
        if isinstance(value, dict):
            # Need to handle _value_
            if hasattr(value, "_value_"):
                dict_to_fill[full_qualified_key] = to_str_dict_format(value._value_)
            extract_keys_from_dict(dict_to_fill, value, full_qualified_key)
        else:
            dict_to_fill[full_qualified_key] = to_str_dict_format(value)


# -----------------------------------------------------------------------------


def flatten_hierarchical_map(hierarchical_map):
    """Helper function that take a hierarchical map and return a flat
    version of it.
    """
    flat_map = {}
    extract_keys_from_dict(flat_map, hierarchical_map, parent_namespace="")
    return flat_map


# -----------------------------------------------------------------------------
# Main DB Object
# -----------------------------------------------------------------------------


class PFDBObj:
    """Core ParFlow Database Object node"""

    def __init__(self, parent=None):
        """
        Create container object while keeping a reference to your parent
        """
        self._parent_ = parent
        self._prefix_ = None

    # -------------------------------------------------------------------------

    def __setitem__(self, key, value):
        """Allow a[x] for assignment as well"""
        self.__setattr__(key, value)

    # -------------------------------------------------------------------------

    def __setattr__(self, name, value):
        """
        Helper method that aims to streamline dot notation assignment
        """
        domains = None
        handlers = None
        history = None
        value_object_assignment = False
        if is_not_private_key(name) and hasattr(self, "_details_"):
            if name in self._details_:
                domains, handlers, history, crosscheck = detail_helper(
                    self, name, value
                )
            elif hasattr(self, name) and isinstance(self.__dict__[name], PFDBObj):
                # Handle value object assignment
                value_object_assignment = True
                value_obj = self.__dict__[name]
                domains, handlers, history, crosscheck = detail_helper(
                    value_obj, "_value_", value
                )
            else:
                msg = (
                    f"{self.full_name()}: Field {name} is not part of the "
                    f"expected schema {self.__class__}"
                )
                print(msg)
                if settings.EXIT_ON_ERROR:
                    raise ValueError(msg)

        # Run domain validation
        if settings.PRINT_LINE_ERROR:
            validate_value_with_exception(value, domains, settings.EXIT_ON_ERROR)

        if value_object_assignment:
            self.__dict__[name].__dict__["_value_"] = decorate_value(
                value, self, handlers
            )
        else:
            # Decorate value if need be (i.e. Geom.names: 'a b c')
            self.__dict__[name] = decorate_value(value, self, handlers)

    # -------------------------------------------------------------------------

    def __len__(self):
        """
        Return the count of nested fields.
          - If a field is not set but is Mandatory it will count as 1
          - If a field is not set, it will count as 0
          - A container does not count. (0)
        """
        value_count = 0

        if hasattr(self, "_value_") and self._value_ is not None:
            value_count += 1

        for name in self.keys(True):
            obj = self.__dict__[name]
            if isinstance(obj, PFDBObj):
                value_count += len(obj)
            elif obj is not None:
                value_count += 1
            elif (
                hasattr(self, "_details_")
                and name in self._details_
                and "domains" in self._details_[name]
            ):
                if "MandatoryValue" in self._details_[name]["domains"]:
                    value_count += 1

        return value_count

    # -------------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Used for obj[] lookup:
           - Need to handle key with prefix
           - Need to handle key with missing prefix
           - Need to handle int key
        """
        key_str = str(key)

        if hasattr(self, key_str):
            return getattr(self, key_str, None)

        prefix = ""
        if hasattr(self, "_details_") and "_prefix_" in self._details_:
            prefix = self._details_["_prefix_"]

        key_str = f"{prefix}{key_str}"
        if not hasattr(self, key_str):
            print(
                f"Could not find key {key}/{key_str} in "
                f"{list(self.__dict__.keys())}"
            )

        return getattr(self, key_str, None)

    # -------------------------------------------------------------------------

    def to_dict(self):
        """Method that will return a flat map of all the ParFlow keys.

        Returns:
          dict: Return Python dict with all the key set listed without
              any hierarchy.
        """
        key_dict = {}
        extract_keys_from_object(key_dict, self)
        return key_dict

    # -------------------------------------------------------------------------

    def keys(self, skip_default=False):
        """
        Gets the key names necessary for the run while skiping unset ones
        """
        for name in self.__dict__:
            if name is None:
                print("need to fix the children instantiator")
                continue

            if is_private_key(name):
                continue

            obj = self.__dict__[name]
            if isinstance(obj, PFDBObj):
                if len(obj):
                    yield name

            else:
                has_details = hasattr(self, "_details_") and name in self._details_
                details = self._details_[name] if has_details else None
                has_default = has_details and "default" in details
                has_domain = has_details and "domains" in details
                is_ignored = has_details and "ignore" in details
                is_mandatory = has_domain and "MandatoryValue" in details["domains"]
                is_default = has_default and obj == details["default"]
                is_set = has_details and details.get("history")

                if is_ignored:
                    continue

                if obj is not None:
                    if skip_default:
                        if not is_default or is_mandatory or is_set:
                            yield name
                    else:
                        yield name

                elif is_mandatory:
                    yield name

    # -------------------------------------------------------------------------

    def validate(
        self, indent=1, verbose=False, enable_print=True, working_directory=None
    ):
        """
        Method to validate sub hierarchy
        """
        if len(self) == 0:
            return 0

        # overwrite current working directory
        prev_dir = settings.WORKING_DIRECTORY
        if working_directory:
            settings.set_working_directory(working_directory)

        error_count = 0
        indent_str = "  " * indent
        for name in self.keys(skip_default=True):
            obj = self.__dict__[name]
            if isinstance(obj, PFDBObj):
                if len(obj):
                    if hasattr(obj, "_value_"):
                        value = obj._value_
                        add_errors, validation_string = validate_helper(
                            obj, "_value_", value, indent
                        )

                        if enable_print and (add_errors or verbose):
                            print(f"{indent_str}{name}: {validation_string}")

                        error_count += add_errors
                    elif enable_print:
                        if verbose or obj.validate(enable_print=False):
                            print(f"{indent_str}{name}:")

                    error_count += obj.validate(
                        indent + 1, verbose=verbose, enable_print=enable_print
                    )

            elif hasattr(self, "_details_") and name in self._details_:
                add_errors, validation_string = validate_helper(self, name, obj, indent)
                if enable_print and (verbose or add_errors):
                    print(f"{indent_str}{name}: {validation_string}")
                error_count += add_errors
            elif obj is not None:
                if enable_print and verbose:
                    print(f"{indent_str}{name}: {obj}")

        # revert working directory to original directory
        settings.set_working_directory(prev_dir)

        return error_count

    # -------------------------------------------------------------------------

    def full_name(self):
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
                if value is current_location:
                    prefix = current_location._prefix_
                    full_path.append(remove_prefix(name, prefix))
            current_location = parent
            if count > len(full_path):
                return f'not found {count}: {".".join(full_path)}'

        full_path.reverse()
        return ".".join(full_path)

    # -------------------------------------------------------------------------

    def to_pf_name(self, parent_namespace, key):
        """
        Helper method returning the key to use for Parflow on a given
        field key. This allows us to handle differences between what
        can be defined in Python vs Parflow key.
        """
        value = self.__dict__[key]
        prefix = ""
        if isinstance(value, PFDBObj):
            if value._prefix_ and key.startswith(value._prefix_):
                prefix = value._prefix_
        elif key in self._details_:
            detail = self._details_[key]
            if "_prefix_" in detail:
                prefix = detail["_prefix_"]

        start = f"{parent_namespace}." if parent_namespace else ""
        return start + remove_prefix(key, prefix)

    # -------------------------------------------------------------------------

    def get_children_of_type(self, class_name):
        """Return a list of PFDBObj of a given type that are part of
        our children.
        """
        results = []
        for key, value in self.__dict__.items():
            if is_private_key(key):
                continue
            if value.__class__.__name__ == class_name:
                results.append(value)

        return results

    # -------------------------------------------------------------------------

    @normalize_location
    def select(self, location="."):
        """
        Return a PFDBObj object based on a location.

        i.e.:
          run.Process.Topology.select('.') => run.Process.Topology
          run.Process.Topology.select('..') => run.Process
          run.Process.Topology.select('../../Geom') => run.Geom
          run.Process.Topology.select('/Geom') => run.Geom
        """
        current_location = self
        path_items = location.split("/")
        if location.startswith("/"):
            while current_location._parent_ is not None:
                current_location = current_location._parent_

        next_list = [current_location]
        for path_item in path_items:
            if not path_item:
                continue

            current_list = next_list
            next_list = []

            if path_item == "..":
                next_list.extend(map(map_to_parent, current_list))
            elif path_item == ".":
                next_list.extend(map(map_to_self, current_list))
            elif path_item.startswith("{"):
                multi_list = map(map_to_children_of_type(path_item[1:-1]), current_list)
                next_list = [x for sublist in multi_list for x in sublist]
            else:
                next_list.extend(
                    filter(filter_none, map(map_to_child(path_item), current_list))
                )
                if next_list and isinstance(next_list[0], list):
                    next_list = [x for sublist in next_list for x in sublist]

        return next_list

    # -------------------------------------------------------------------------

    @normalize_location
    def value(self, location=".", skip_default=False):
        """
        Return a value based on a location.

        e.g.:
          run.ComputationalGrid.value('DX') => 8.88
          run.value('ComputationalGrid/DX') => 8.88
          run.value('/ComputationalGrid/DX') => 8.88
          run.Perm.value('../ComputationalGrid/DX') => 8.88
          run.Solver.value() => 'Richards'
        """
        value, *_ = self._value(location)

        if skip_default:
            details = self.details(location)
            if not details.get("history"):
                return

        return value

    # -------------------------------------------------------------------------

    @normalize_location
    def details(self, location="."):
        """
        Return details based on a location.
        """
        value, container, key = self._value(location)

        if key and isinstance(getattr(container, key, None), PFDBObj):
            value = container[key]

        if isinstance(value, PFDBObj):
            return getattr(value, "_details_", {}).get("_value_", {})

        key = "_value_" if key is None else key
        return getattr(container, "_details_", {}).get(key, {})

    # -------------------------------------------------------------------------

    @normalize_location
    def doc(self, location="."):
        """
        Return docs based on a location.
        """
        value, container, key = self._value(location)
        details = self.details(location)

        if key and isinstance(getattr(container, key, None), PFDBObj):
            value = container[key]

        if not value and isinstance(container, PFDBObj):
            value = container

        ret = ""
        if isinstance(value, PFDBObj) and getattr(value, "__doc__", None):
            ret += value.__doc__ + "\n"
        if details.get("help"):
            ret += details["help"] + "\n"
        return ret

    # -------------------------------------------------------------------------

    def get_context_settings(self):
        """
        Return global settings for our current parflow run.
        This is useful when providing global information for domains or else.
        """
        return {
            "print_line_error": settings.PRINT_LINE_ERROR,
            "exit_on_error": settings.EXIT_ON_ERROR,
            "working_directory": settings.WORKING_DIRECTORY,
            "pf_version": settings.PARFLOW_VERSION,
        }

    # ---------------------------------------------------------------------------

    def pfset(
        self,
        key="",
        value=None,
        yaml_file=None,
        yaml_content=None,
        pfidb_file=None,
        hierarchical_map=None,
        flat_map=None,
        exit_if_undefined=False,
        silence_if_undefined=False,
    ):
        """
        Allow to define any parflow key so it can be exported.
        Many formats are supported:
            - key/value: To set a single value relative to our current
                PFDBObj.
            - yaml_file: YAML file path to load and import using the
                current PFDBObj as root.
            - yaml_content: YAML string to load and import using the
                current PFDBObj as root.
            - hierarchical_map: Nested dict containing several key/value
                pair using the current PFDBObj as root.
            - flat_map: Flat dict with parflow key/value pair to set
                using the current PFDBObj as root.
        """
        if yaml_file:
            yaml_content = get_text_file_content(yaml_file)

        if yaml_content:
            hierarchical_map = yaml.safe_load(yaml_content)

        if pfidb_file:
            flat_map = read_pfidb(pfidb_file)

        if hierarchical_map:
            flat_map = flatten_hierarchical_map(hierarchical_map)

        if flat_map:
            sorted_flat_map = sort_dict_by_priority(flat_map)
            for key, value in sorted_flat_map.items():
                self.pfset(
                    key=key,
                    value=value,
                    exit_if_undefined=exit_if_undefined,
                    silence_if_undefined=silence_if_undefined,
                )

        if not key:
            return

        key_stored = False
        tokens = key.split(".")
        if len(tokens) > 1:
            value_key = tokens[-1]
            container = None
            query = "/".join(tokens[:-1])
            selection = self.select(query)

            if len(selection) > 0:
                container = selection[0]

            if len(selection) > 1:
                raise ValueError(
                    f"Found {len(selection)} containers when selecting {query}: expected one or zero"
                )

            if container is None:
                # We need to maybe handle prefix
                value_key = tokens.pop()
                container = self
                for name in tokens:
                    if container is None:
                        break

                    if name in container.__dict__:
                        container = container[name]
                    else:
                        # Extract available prefix
                        known_prefixes = set("_")
                        for child_name in container.keys():
                            if isinstance(container[child_name], PFDBObj):
                                prefix = getattr(container[child_name], "_prefix_", "")
                                if prefix is not None:
                                    known_prefixes.add(prefix)

                        found = False

                        # Test names with prefix
                        for prefix in known_prefixes:
                            if found:
                                break
                            name_w_prefix = f"{prefix}{name}"
                            if name_w_prefix in container.__dict__:
                                found = True
                                container = container[name_w_prefix]

                        # No matching prefix
                        if not found:
                            container = None

            if container is not None:
                container[value_key] = value
                key_stored = True
        elif len(tokens) == 1:
            self[tokens[0]] = value
            key_stored = True

        if not key_stored:
            # Only create a store at the root node
            root = self
            while root._parent_ is not None:
                root = root._parent_

            # store key on the side
            if "_pfstore_" not in root.__dict__:
                root.__dict__["_pfstore_"] = {}
            parent_namespace = self.full_name()
            full_key_name = (
                f"{parent_namespace}" f"{'.' if parent_namespace else ''}{key}"
            )
            root.__dict__["_pfstore_"][full_key_name] = value
            root_path = self.full_name()
            if not silence_if_undefined:
                print(
                    f"Caution: Using internal store of "
                    f"{root_path if root_path else 'run'} "
                    f"to save {full_key_name} = {value}"
                )
            if exit_if_undefined:
                sys.exit(1)

    # ---------------------------------------------------------------------------

    def _process_dynamic(self):
        """
        Processing the dynamically defined (user-defined) key names
        """
        from . import generated

        for class_name, selection in self._dynamic_.items():
            klass = getattr(generated, class_name)
            names = self.select(selection)
            for name in names:
                if name is not None:
                    self.__dict__[name] = klass(self)

    # -------------------------------------------------------------------------

    @normalize_location
    def _value(self, location="."):
        """
        Internal function to get the value, container and key from location
        """
        container, key = self._get_container_and_key(location)

        if key is None:
            value = container
        else:
            value = getattr(container, key, None)

        if isinstance(value, PFDBObj):
            value = getattr(value, "_value_", None)

        # Try to do a store lookup
        if value is None or container is None:
            root = container if container else self
            while root._parent_ is not None:
                root = root._parent_

            if container is not None and key is not None:
                full_key_name = ".".join([container.full_name(), key])
                if "_pfstore_" in root.__dict__:
                    store = root.__dict__["_pfstore_"]
                    if full_key_name in store:
                        return store[full_key_name], root, full_key_name

            # no container were found need to start from root
            path_tokens = location.split("/")

            if len(path_tokens[0]) == 0:
                # We have abs_path
                full_key_name = ".".join(path_tokens[1:])
            elif len(path_tokens):
                # relative path
                local_root = self
                while len(path_tokens) and path_tokens[0] == ".":
                    path_tokens.pop(0)
                while len(path_tokens) and path_tokens[0] == "..":
                    local_root = local_root._parent_
                    path_tokens.pop(0)

                prefix_name = local_root.full_name()
                if prefix_name:
                    path_tokens.insert(0, prefix_name)

                full_key_name = ".".join(path_tokens)

            # Resolve full_key_name from root
            current_key = full_key_name
            current_node = root
            current_store = getattr(current_node, "_pfstore_", None)

            # Find key in store
            while len(current_key) and current_store is not None:
                if current_key in current_store:
                    return current_store[current_key], root, current_key

                # Find child store
                current_store = None
                while (
                    len(current_key) > 0
                    and current_node is not None
                    and current_store is None
                ):
                    tokens = current_key.split(".")
                    current_node = current_node[tokens[0]]
                    current_store = getattr(current_node, "_pfstore_", None)
                    current_key = ".".join(tokens[1:])

        return value, container, key

    # -------------------------------------------------------------------------

    @normalize_location
    def _get_container_and_key(self, location):
        """
        Internal function to get the container and key from location
        """
        split = location.split("/")
        path, key = "/".join(split[:-1]), split[-1]

        if key == ".":
            return self, None

        parent = self
        while key == ".." and getattr(parent, "_parent_", None):
            parent = parent._parent_
            split = path.split("/")
            path, key = "/".join(split[:-1]), split[-1]

        if parent is not self:
            # We went through the loop above. Return...
            return parent, None

        selection = self.select(path)
        if len(selection) == 0:
            return None, None

        return selection[0], key


# -----------------------------------------------------------------------------
# Main DB Object
# -----------------------------------------------------------------------------


class PFDBObjListNumber(PFDBObj):
    """Class for leaf list values"""

    def __setattr__(self, name, value):
        """Helper method that aims to streamline dot notation assignment"""
        key_str = str(name)
        if is_private_key(key_str):
            self.__dict__[key_str] = value
            return

        if self._prefix_:
            if key_str.startswith(self._prefix_):
                self.__dict__[key_str] = value
            else:
                self.__dict__[f"{self._prefix_}{key_str}"] = value
            return

        self.__dict__[key_str] = value

    def to_pf_name(self, parent_namespace, key):
        """Helper method returning the key to use for Parflow on
        a given field key. This allows handling of differences
        between what can be defined in Python vs Parflow key.
        """
        start = f"{parent_namespace}." if parent_namespace else ""
        return start + remove_prefix(key, self._prefix_)
