r'''
This module aims to gather all kind of value handler you would like to
enable inside Parflow run.

A value handler is responsible to process the user input and returned
a possibly modified version of it while maybe affecting that container
object along the way.
'''

import sys
from . import generated
from ..terminal import Colors as term
from ..terminal import Symbols as term_symbol

# -----------------------------------------------------------------------------


class ValueHandlerException(Exception):
    '''
    Basic parflow exception used for ValueHandlers to report error
    '''
    pass

# -----------------------------------------------------------------------------


class ChildrenHandler:
    '''
    This class takes creates new keys from user-defined name inputs (e.g. GeomNames)
    '''
    def decorate(self, value, container, class_name=None, location='.', eager=None, **kwargs):
        klass = getattr(generated, class_name)
        destination_containers = container.get_selection_from_location(location)

        if isinstance(value, str):
            names = value.split(' ')
            valid_names = []
            for name in names:
                if len(name):
                    # print(f' - {name} => {class_name} in {destination_container.__class__}')
                    valid_names.append(name)
                    for destination_container in destination_containers:
                        if destination_container is not None:
                            if name not in destination_container.__dict__:
                                destination_container.__dict__[
                                    name] = klass(destination_container)
                        elif eager:
                            print(f'Error no selection for {location}')

            return valid_names

        # for handling variable DZ setting and BCPressure/BCSaturation NumPoints (and possibly others)
        elif isinstance(value, int):
            valid_names = []
            for i in range(value):
                name = f'_{i}'
                valid_names.append(name)
                for destination_container in destination_containers:
                    if destination_container is not None:
                        if name not in destination_container.__dict__:
                            destination_container.__dict__[
                                name] = klass(destination_container)
                    elif eager:
                        print(f'Error no selection for {location}')

            return valid_names

        if hasattr(value, '__iter__'):
            valid_names = []
            for name in value:
                if len(name):
                    valid_names.append(name)
                    # print(f' - {name} => {class_name} in {destination_container.__class__}')
                    for destination_container in destination_containers:
                        destination_container.__dict__[
                            name] = klass(destination_container)

            return valid_names

        raise ValueHandlerException(
            f'{value} is not of the expected type for GeometryNameHandler')

# -----------------------------------------------------------------------------
# Helper map with an instance of each Value handler
# -----------------------------------------------------------------------------


AVAILABLE_HANDLERS = {}


def get_handler(class_name, print_error=True):
    if class_name in AVAILABLE_HANDLERS:
        return AVAILABLE_HANDLERS[class_name]

    if hasattr(sys.modules[__name__], class_name):
        klass = getattr(sys.modules[__name__], class_name)
        instance = klass()
        AVAILABLE_HANDLERS[class_name] = instance
        return instance

    if print_error:
        print(
            f'{term.FAIL}{term_symbol.ko}{term.ENDC} Could not find handler: "{class_name}"')

    return None

# -----------------------------------------------------------------------------
# API meant to be used outside of this module
# -----------------------------------------------------------------------------


def decorate_value(value, container=None, handlers=None):
    '''
    handlers = {
        GeomInputUpdater: {
          type: 'ChildrenHandler',
          className: 'GeomInputItemValue',
          location: '../..'
        },
        GeomUpdater: {
          type: 'ChildrenHandler',
          className: 'GeomItem',
          location: '../../Geom'
        },
        ChildrenHandler: {
          className: 'GeomInputLocal'
        }
    }
    '''
    if handlers is None:
        return value

    return_value = value

    for handler_classname in handlers:
        handler = get_handler(handler_classname, False)

        if not handler and 'type' in handlers[handler_classname]:
            handler = get_handler(handlers[handler_classname]['type'])
        else:
            get_handler(handler_classname)

        if handler:
            handler_kwargs = handlers[handler_classname]
            if isinstance(handler_kwargs, str):
                return_value = handler.decorate(value, container)
            else:
                return_value = handler.decorate(
                    value, container, **handler_kwargs)

    # added to handle variable DZ
    if isinstance(value, int):
        return value

    return return_value
