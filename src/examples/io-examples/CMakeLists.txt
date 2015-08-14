if(WITH_HDF5 AND HDF5_FOUND)
   add_executable(example-io io_graphical_model.cxx ${headers})
   if(WIN32 OR APPLE)
      target_link_libraries(example-io ${HDF5_LIBRARIES})
   else()
      target_link_libraries(example-io ${HDF5_LIBRARIES} rt)
   endif()

   add_executable(example-io-benchmark io_benchmark.cxx)

   target_link_libraries(example-io-benchmark ${CMAKE_THREAD_LIBS_INIT}
                                              ${CPLEX_LIBRARIES}
                                              ${HDF5_LIBRARIES}
                                              rt
                                              boost_system
                                              boost_chrono
                                              part_opt
                                              exttype
                                              maxflow_BK
                                              external-library-qpbo)
endif()
