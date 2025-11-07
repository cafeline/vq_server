cmake_minimum_required(VERSION 3.8)
project(vq_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(vq_server src/vq_server.cpp)
target_include_directories(vq_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(vq_server PRIVATE ${HDF5_INCLUDE_DIRS})
target_compile_features(vq_server PUBLIC cxx_std_17)

ament_target_dependencies(vq_server
  rclcpp
  visualization_msgs
  geometry_msgs
  ament_index_cpp
)

target_link_libraries(vq_server
  ${HDF5_C_LIBRARIES}
  ${HDF5_HL_LIBRARIES}
)

install(TARGETS vq_server
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  config
  launch
  maps
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
