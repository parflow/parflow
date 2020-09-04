# -----------------------------------------------------------------------------
# Map function Helper functions
# -----------------------------------------------------------------------------

def map_to_parent(pfdbObj):
    """Helper function to extract the parent of a pfdbObj"""
    return pfdbObj._parent_

# -----------------------------------------------------------------------------

def map_to_self(pfdbObj):
    """Helper function to extract self of self (noop)"""
    return pfdbObj

# -----------------------------------------------------------------------------

def map_to_child(name):
    """Helper function that return a function for extracting a field name
    """
    return lambda pfdbObj: getattr(pfdbObj, name) if hasattr(pfdbObj, name) else None

# -----------------------------------------------------------------------------

def map_to_children_of_type(class_name):
    """Helper function that return a function for extracting children
    of a given type (class_name).
    """
    def get_children_of_type(pfdbObj):
        return pfdbObj.get_children_of_type(class_name)
    return get_children_of_type

# -----------------------------------------------------------------------------
# Key dictionary helpers
# -----------------------------------------------------------------------------

def get_key_priority(key_name):
    """Return number that can be used to sort keys in term of priority
    """
    priority_value = 0
    path_token = key_name.split('.')
    if 'Name' in key_name:
        priority_value -= 100

    for token in path_token:
        if token[0].isupper():
            priority_value += 1
        else:
            priority_value += 10

    priority_value *= 100
    priority_value += len(key_name)

    return priority_value

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Sort helpers
# -----------------------------------------------------------------------------

def sort_dict(input):
    """Create a key sorted dict
    """
    output = {}
    keys = list(input.keys())
    keys.sort()
    for key in keys:
        output[key] = input[key]

    return output

# -----------------------------------------------------------------------------

def sort_dict_by_priority(input):
    """Create a key sorted dict
    """
    key_list = []
    for key, value in input.items():
        key_list.append((key, value, get_key_priority(key)))

    key_list.sort(key=lambda t: t[2])

    output = {}
    for item in key_list:
        output[item[0]] = item[1]

    return output

# -----------------------------------------------------------------------------
# Dictionary helpers
# -----------------------------------------------------------------------------

def get_or_create_dict(root, keyPath, overriden_keys):
    """Helper function to get/create a container dict for a given key path
    """
    currentContainer = root
    for i in range(len(keyPath)):
        if keyPath[i] not in currentContainer:
            currentContainer[keyPath[i]] = {}
        elif not isinstance(currentContainer[keyPath[i]], dict):
            overriden_keys['.'.join(keyPath[:i+1])
                           ] = currentContainer[keyPath[i]]
            currentContainer[keyPath[i]] = {}
        currentContainer = currentContainer[keyPath[i]]

    return currentContainer

# -----------------------------------------------------------------------------
