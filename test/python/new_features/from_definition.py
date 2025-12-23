# -----------------------------------------------------------------------------
# Testing pfset with a .pfidb file
# -----------------------------------------------------------------------------

from pathlib import Path

from parflow import Run
from parflow.tools.fs import get_absolute_path

dsingle = Run("dsingle", __file__)

dsingle.pfset(
    pfidb_file=get_absolute_path("$PF_SRC/test/correct_output/dsingle.pfidb.ref")
)


# Test pfidb
generated, _ = dsingle.write()
old_text = Path(generated).read_text()

dsingle2 = Run.from_definition(generated)
generated, _ = dsingle2.write()

new_text = Path(generated).read_text()

assert old_text and new_text
assert old_text == new_text


# Test yaml
generated, _ = dsingle.write(file_format="yaml")
old_text = Path(generated).read_text()

# Make sure we are not repeating ourselves
assert old_text != new_text

dsingle2 = Run.from_definition(generated)
generated, _ = dsingle2.write(file_format="yaml")

new_text = Path(generated).read_text()

assert old_text and new_text
assert old_text == new_text
