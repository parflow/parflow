set(SRC_FILES cJSON.c)

add_library(cjson STATIC ${SRC_FILES})

install(TARGETS cjson EXPORT ParFlow DESTINATION lib)




