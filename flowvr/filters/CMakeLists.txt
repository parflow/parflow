INCLUDE(CheckFunctionExists)
INCLUDE(CheckSymbolExists)

# plugins
SET(plugins-filters-SRC flowvr.plugins.MergeItExt.cpp)

FOREACH(type sys sync filters)
  FOREACH(plugin-src ${plugins-${type}-SRC})
    STRING(REPLACE ".cpp" "" plugin ${plugin-src})
    ADD_LIBRARY(${plugin} MODULE ${plugin-src})
    SET_TARGET_PROPERTIES(${plugin} PROPERTIES PREFIX "")
    TARGET_LINK_LIBRARIES(${plugin} flowvr-base flowvr-plugd)
    INSTALL(TARGETS ${plugin} DESTINATION ${FLOWVR_PREFIX}/lib/flowvr/plugins)
  ENDFOREACH(plugin-src)
ENDFOREACH(type)

