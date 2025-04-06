# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-src")
  file(MAKE_DIRECTORY "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-src")
endif()
file(MAKE_DIRECTORY
  "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-build"
  "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-subbuild/googletest-populate-prefix"
  "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-subbuild/googletest-populate-prefix/tmp"
  "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
  "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-subbuild/googletest-populate-prefix/src"
  "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/furkan/projects/school/mmi713/evm/evmcpp/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
