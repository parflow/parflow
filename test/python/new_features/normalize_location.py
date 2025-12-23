# -----------------------------------------------------------------------------
# Testing pfset with a .pfidb file
# -----------------------------------------------------------------------------

from parflow.tools.helper import normalize_location


@normalize_location
def new_path(x):
    return x


assert new_path("") == "."
assert new_path(".path") == "path"
assert new_path("path.to") == "path/to"
assert new_path(".path.to.the.obj") == "path/to/the/obj"
assert new_path(".path/to/the.obj") == "path/to/the/obj"
assert new_path("./path/to/the/obj") == "path/to/the/obj"
assert new_path("./path/to/the//obj/../../to/././the") == "path/to/to/the"

# These should be no-op
no_op_strings = [
    ".",
    ".." "path",
    "/path/to/the/obj",
    "path/to/the/obj",
]

for s in no_op_strings:
    assert new_path(s) == s
