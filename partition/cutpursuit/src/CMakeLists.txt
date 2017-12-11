set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -fopenmp -O3 -Wall -std=c++11")

add_library(cp SHARED cutpursuit.cpp ${CP_HEADERS})
target_link_libraries(cp
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    boost_python
    )
