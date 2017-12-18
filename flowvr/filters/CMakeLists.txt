INCLUDE(CheckFunctionExists)
INCLUDE(CheckSymbolExists)

# plugins
SET(plugins-filters-SRC flowvr.plugins.MergeItExt.cpp)

FOREACH(type sys sync filters)
  FOREACH(plugin-src ${plugins-${type}-SRC})
    STRING(REPLACE ".cpp" "" plugin ${plugin-src})
    ADD_LIBRARY(${plugin} MODULE ${plugin-src})
    TARGET_INCLUDE_DIRECTORIES(${plugin} PUBLIC ${FLOWVR_base_INCLUDE_DIR}
${FLOWVR_plugd_INCLUDE_DIR})
    SET_TARGET_PROPERTIES(${plugin} PROPERTIES PREFIX "")
    TARGET_LINK_LIBRARIES(${plugin} ${FLOWVR_base_LIBRARY}
${FLOWVR_plugd_LIBRARY})
    INSTALL(TARGETS ${plugin} DESTINATION ${FLOWVR_PREFIX}/lib/flowvr/plugins)
  ENDFOREACH(plugin-src)
ENDFOREACH(type)

