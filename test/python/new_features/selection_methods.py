# -----------------------------------------------------------------------------
# Testing various methods that use locations for selections
# -----------------------------------------------------------------------------

from parflow import Run
from parflow.tools.fs import get_absolute_path

run = Run("run", __file__)

run.pfset(pfidb_file=get_absolute_path("$PF_SRC/test/correct_output/dsingle.pfidb.ref"))

# Some modifications for testing
run.Solver = "Richards"

# select() tests
porosity = run.Geom.Porosity
assert run.Geom.Porosity.select(".")[0] is porosity
assert porosity.select(".")[0] is porosity
assert porosity.select()[0] is porosity
assert porosity.select("..")[0] is run.Geom
assert porosity.select("GeomNames") == porosity.GeomNames
assert porosity.select("./GeomNames") == porosity.GeomNames
assert run.Geom.select("Porosity/GeomNames") == porosity.GeomNames
assert porosity.select("/")[0] is run
assert porosity.select("/Geom")[0] is run.Geom
assert porosity.select("/Geom/Porosity")[0] is porosity
assert porosity.select("../../Solver/../Geom/./Porosity/..")[0] is run.Geom
assert porosity.select("../..")[0] is run
assert run.Geom.select(".Porosity.GeomNames") == porosity.GeomNames

# value() tests
background = run.Geom.background
assert run.value("Geom/background/Porosity/Value") == run.Geom.background.Porosity.Value
assert (
    run.value("/Geom/background/Porosity/Value") == run.Geom.background.Porosity.Value
)
assert (
    background.value("/Geom/background/Porosity/Value")
    == run.Geom.background.Porosity.Value
)
assert (
    background.value("tce/Retardation/Type") == run.Geom.background.tce.Retardation.Type
)
assert (
    background.value("./tce/Retardation/Type")
    == run.Geom.background.tce.Retardation.Type
)
assert (
    background.value("tce/Retardation/Rate") == run.Geom.background.tce.Retardation.Rate
)
assert background.value("../concen_region/Lower/Z") == run.Geom.concen_region.Lower.Z
assert background.value("../domain/Patches") == run.Geom.domain.Patches
assert run.Solver.value(".") == "Richards"
assert run.Solver.value() == "Richards"
assert run.Solver.CLM.value("..") == "Richards"
assert run.value("./UseClustering") == run.UseClustering
assert run.value("./UseClustering") is not None
assert run.value("./UseClustering", skip_default=True) is None
assert run.value("./Geom/background/../.././Solver") == "Richards"
assert run.value("Geom.background.Porosity.Value") == run.Geom.background.Porosity.Value
assert run.value("Cell/_0/dzScale/Value") == run.Cell._0.dzScale.Value
assert run.value("./Cell/_0/dzScale/Value") == run.Cell._0.dzScale.Value
assert run.Cell.value("_0/dzScale/Value") == run.Cell._0.dzScale.Value
assert run.Cell._0.dzScale.value("Value") == run.Cell._0.dzScale.Value

# details() tests
lower = run.Geom.background.Lower
upper = run.Geom.background.Upper
assert lower.details("X") is lower["_details_"]["X"]
assert lower.details("./X") is lower["_details_"]["X"]
assert lower.details("../Upper/Y") is upper["_details_"]["Y"]
assert run.Solver.details(".") is run.Solver["_details_"]["_value_"]
assert (
    run.Solver.details(".").get("default")
    == run.Solver["_details_"]["_value_"]["default"]
)
assert run.Solver.CLM.details("..") is run.Solver["_details_"]["_value_"]
assert (
    run.Solver.CLM.details("..").get("default")
    == run.Solver["_details_"]["_value_"]["default"]
)
assert (
    run.details("Solver/TerrainFollowingGrid").get("default")
    == run.Solver.TerrainFollowingGrid["_details_"]["_value_"]["default"]
)
assert (
    run.details("/Solver/TerrainFollowingGrid").get("default")
    == run.Solver.TerrainFollowingGrid["_details_"]["_value_"]["default"]
)
assert (
    run.details("Solver.TerrainFollowingGrid").get("default")
    == run.Solver.TerrainFollowingGrid["_details_"]["_value_"]["default"]
)

# doc() tests
assert run.doc("/Domain") == run.Domain.__doc__ + "\n"
assert run.doc("Domain") == run.Domain.__doc__ + "\n"
assert run.doc("./Domain") == run.Domain.__doc__ + "\n"
assert run.Domain.doc(".") == run.Domain.__doc__ + "\n"
assert run.Domain.doc() == run.Domain.__doc__ + "\n"
assert (
    run.Domain.doc("../Solver")
    == run.Solver.__doc__ + "\n" + run.Solver._details_["_value_"]["help"] + "\n"
)
assert background.doc(".") == background.__doc__ + "\n"
assert background.doc("./Lower/X") == background.Lower._details_["X"]["help"] + "\n"
assert (
    background.doc("./Lower/../..//./background/Upper/Y/.")
    == background.Upper._details_["Y"]["help"] + "\n"
)
assert background.Upper.doc("../..") == run.Geom.__doc__ + "\n"
assert background.doc(".Lower.X") == background.Lower._details_["X"]["help"] + "\n"
