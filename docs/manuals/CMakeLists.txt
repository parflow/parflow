INCLUDE(UseLATEX)

ADD_LATEX_DOCUMENT(
    user_manual.tex
    INPUTS parflow.sty defs.tex inputps.tex intro.tex start.tex pfsystem.tex pftools.tex models.tex files.tex fdl.tex bib.tex
    IMAGES IGWMC-logo-new.pdf
    USE_INDEX
    )

ADD_LATEX_DOCUMENT(
    dev_manual.tex
    INPUTS parflow.sty defs.tex dev_manual_intro.tex codeorg.tex documentation.tex modules.tex input.tex equations.tex grid.tex discretize.tex visualization.tex administrative.tex ref.tex ref.grid.tex
    )

ADD_LATEX_DOCUMENT(
     amps_manual.tex
    INPUTS parflow.sty
    )

configure_file (version.tex.in version.tex)
