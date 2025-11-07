#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace
{
struct CompressedMap
{
  float voxel_size = 0.1f;
  uint32_t block_size = 8;
  uint32_t pattern_length = 0;
  uint8_t block_index_bit_width = 0;

  std::array<double, 3> grid_origin{{0.0, 0.0, 0.0}};
  std::array<int32_t, 3> block_dims{{0, 0, 0}};
  std::array<int32_t, 3> block_offset{{0, 0, 0}};

  std::vector<uint8_t> dictionary_patterns;
  std::vector<uint32_t> block_indices;

  uint32_t sentinel_value = std::numeric_limits<uint32_t>::max();
  std::string frame_id{"map"};
};

class H5ObjectHandle
{
public:
  explicit H5ObjectHandle(hid_t id = -1) : id_(id) {}
  ~H5ObjectHandle()
  {
    close();
  }
  H5ObjectHandle(const H5ObjectHandle &) = delete;
  H5ObjectHandle & operator=(const H5ObjectHandle &) = delete;
  H5ObjectHandle(H5ObjectHandle && other) noexcept : id_(std::exchange(other.id_, -1)) {}
  H5ObjectHandle & operator=(H5ObjectHandle && other) noexcept
  {
    if (this != &other) {
      close();
      id_ = std::exchange(other.id_, -1);
    }
    return *this;
  }
  hid_t get() const { return id_; }
  explicit operator bool() const { return id_ >= 0; }

private:
  void close()
  {
    if (id_ >= 0) {
      if (H5Iget_type(id_) == H5I_FILE) {
        H5Fclose(id_);
      } else if (H5Iget_type(id_) == H5I_GROUP) {
        H5Gclose(id_);
      } else if (H5Iget_type(id_) == H5I_DATASET) {
        H5Dclose(id_);
      } else if (H5Iget_type(id_) == H5I_DATASPACE) {
        H5Sclose(id_);
      } else if (H5Iget_type(id_) == H5I_ATTR) {
        H5Aclose(id_);
      } else if (H5Iget_type(id_) == H5I_DATATYPE) {
        H5Tclose(id_);
      } else {
        H5Oclose(id_);
      }
      id_ = -1;
    }
  }

  hid_t id_{-1};
};

std::string read_string_attribute(hid_t group_id, const char * attribute_name)
{
  if (H5Aexists(group_id, attribute_name) <= 0) {
    return {};
  }
  H5ObjectHandle attribute{H5Aopen(group_id, attribute_name, H5P_DEFAULT)};
  if (!attribute) {
    return {};
  }
  H5ObjectHandle type{H5Aget_type(attribute.get())};
  if (!type) {
    return {};
  }
  size_t size = H5Tget_size(type.get());
  if (size == 0) {
    return {};
  }
  std::string buffer(size, '\0');
  if (H5Aread(attribute.get(), type.get(), buffer.data()) < 0) {
    return {};
  }
  auto pos = buffer.find('\0');
  if (pos != std::string::npos) {
    buffer.resize(pos);
  }
  return buffer;
}

bool read_compression_params(const rclcpp::Logger & logger, hid_t file_id, CompressedMap & out)
{
  if (H5Lexists(file_id, "/compression_params", H5P_DEFAULT) <= 0) {
    RCLCPP_ERROR(logger, "HDF5: /compression_params が見つかりません");
    return false;
  }

  H5ObjectHandle group{H5Gopen2(file_id, "/compression_params", H5P_DEFAULT)};
  if (!group) {
    RCLCPP_ERROR(logger, "HDF5: /compression_params のオープンに失敗しました");
    return false;
  }

  auto read_scalar_float = [&](const char * name, float & target) {
    if (H5Lexists(group.get(), name, H5P_DEFAULT) > 0) {
      H5ObjectHandle dataset{H5Dopen2(group.get(), name, H5P_DEFAULT)};
      if (dataset) {
        if (H5Dread(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &target) < 0) {
          RCLCPP_WARN(logger, "HDF5: %s の読み込みに失敗しました", name);
        }
      }
    }
  };

  auto read_scalar_u32 = [&](const char * name, uint32_t & target) {
    if (H5Lexists(group.get(), name, H5P_DEFAULT) > 0) {
      H5ObjectHandle dataset{H5Dopen2(group.get(), name, H5P_DEFAULT)};
      if (dataset) {
        if (H5Dread(dataset.get(), H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &target) < 0) {
          RCLCPP_WARN(logger, "HDF5: %s の読み込みに失敗しました", name);
        }
      }
    }
  };

  read_scalar_float("voxel_size", out.voxel_size);
  read_scalar_u32("block_size", out.block_size);

  uint32_t bit_width_temp = 0;
  read_scalar_u32("block_index_bit_width", bit_width_temp);
  if (bit_width_temp >= 1 && bit_width_temp <= 32) {
    out.block_index_bit_width = static_cast<uint8_t>(bit_width_temp);
  }

  if (H5Lexists(group.get(), "grid_origin", H5P_DEFAULT) > 0) {
    H5ObjectHandle dataset{H5Dopen2(group.get(), "grid_origin", H5P_DEFAULT)};
    if (dataset) {
      float origin[3] = {0.f, 0.f, 0.f};
      if (H5Dread(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, origin) >= 0) {
        out.grid_origin[0] = static_cast<double>(origin[0]);
        out.grid_origin[1] = static_cast<double>(origin[1]);
        out.grid_origin[2] = static_cast<double>(origin[2]);
      } else {
        RCLCPP_WARN(logger, "HDF5: grid_origin の読み込みに失敗しました");
      }
    }
  }

  return true;
}

bool read_metadata(const rclcpp::Logger & logger, hid_t file_id, CompressedMap & out)
{
  if (H5Lexists(file_id, "/metadata", H5P_DEFAULT) <= 0) {
    return true;
  }
  H5ObjectHandle group{H5Gopen2(file_id, "/metadata", H5P_DEFAULT)};
  if (!group) {
    RCLCPP_WARN(logger, "HDF5: /metadata のオープンに失敗しました");
    return true;
  }
  std::string frame = read_string_attribute(group.get(), "frame_id");
  if (!frame.empty()) {
    out.frame_id = frame;
  }
  return true;
}

bool read_dictionary(const rclcpp::Logger & logger, hid_t file_id, CompressedMap & out)
{
  if (H5Lexists(file_id, "/dictionary", H5P_DEFAULT) <= 0) {
    RCLCPP_ERROR(logger, "HDF5: /dictionary が見つかりません");
    return false;
  }
  H5ObjectHandle group{H5Gopen2(file_id, "/dictionary", H5P_DEFAULT)};
  if (!group) {
    RCLCPP_ERROR(logger, "HDF5: /dictionary のオープンに失敗しました");
    return false;
  }

  if (H5Lexists(group.get(), "pattern_length", H5P_DEFAULT) > 0) {
    H5ObjectHandle dataset{H5Dopen2(group.get(), "pattern_length", H5P_DEFAULT)};
    if (dataset) {
      if (H5Dread(
          dataset.get(), H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          &out.pattern_length) < 0)
      {
        RCLCPP_WARN(logger, "HDF5: pattern_length の読み込みに失敗しました");
      }
    }
  }

  if (H5Lexists(group.get(), "patterns", H5P_DEFAULT) <= 0) {
    RCLCPP_ERROR(logger, "HDF5: dictionary/patterns が存在しません");
    return false;
  }

  H5ObjectHandle dataset{H5Dopen2(group.get(), "patterns", H5P_DEFAULT)};
  if (!dataset) {
    RCLCPP_ERROR(logger, "HDF5: dictionary/patterns のオープンに失敗しました");
    return false;
  }

  H5ObjectHandle dataspace{H5Dget_space(dataset.get())};
  if (!dataspace) {
    RCLCPP_ERROR(logger, "HDF5: dictionary/patterns の dataspace 取得に失敗しました");
    return false;
  }

  hssize_t elements = H5Sget_simple_extent_npoints(dataspace.get());
  if (elements <= 0) {
    RCLCPP_WARN(logger, "HDF5: 辞書が空です");
    out.dictionary_patterns.clear();
    return true;
  }

  out.dictionary_patterns.resize(static_cast<size_t>(elements));
  if (H5Dread(
      dataset.get(), H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
      out.dictionary_patterns.data()) < 0)
  {
    RCLCPP_ERROR(logger, "HDF5: dictionary/patterns の読み込みに失敗しました");
    return false;
  }

  return true;
}

bool read_block_dims(
  const rclcpp::Logger & logger, hid_t group_id, const char * name,
  std::array<int32_t, 3> & target)
{
  if (H5Lexists(group_id, name, H5P_DEFAULT) <= 0) {
    RCLCPP_ERROR(logger, "HDF5: %s が見つかりません", name);
    return false;
  }
  H5ObjectHandle dataset{H5Dopen2(group_id, name, H5P_DEFAULT)};
  if (!dataset) {
    RCLCPP_ERROR(logger, "HDF5: %s のオープンに失敗しました", name);
    return false;
  }

  int32_t buffer[3] = {0, 0, 0};
  if (H5Dread(
      dataset.get(), H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) < 0)
  {
    RCLCPP_ERROR(logger, "HDF5: %s の読み込みに失敗しました", name);
    return false;
  }

  target[0] = buffer[0];
  target[1] = buffer[1];
  target[2] = buffer[2];
  return true;
}

uint32_t compute_bitmask(uint8_t bit_width)
{
  if (bit_width == 0) {
    return 0;
  }
  if (bit_width >= 32) {
    return 0xFFFFFFFFu;
  }
  return static_cast<uint32_t>((1ULL << bit_width) - 1ULL);
}

bool read_compressed_data(const rclcpp::Logger & logger, hid_t file_id, CompressedMap & out)
{
  if (H5Lexists(file_id, "/compressed_data", H5P_DEFAULT) <= 0) {
    RCLCPP_ERROR(logger, "HDF5: /compressed_data が見つかりません");
    return false;
  }

  H5ObjectHandle group{H5Gopen2(file_id, "/compressed_data", H5P_DEFAULT)};
  if (!group) {
    RCLCPP_ERROR(logger, "HDF5: /compressed_data のオープンに失敗しました");
    return false;
  }

  if (!read_block_dims(logger, group.get(), "block_dims", out.block_dims)) {
    return false;
  }

  const auto total_blocks = static_cast<int64_t>(out.block_dims[0]) *
    static_cast<int64_t>(out.block_dims[1]) *
    static_cast<int64_t>(out.block_dims[2]);
  if (total_blocks <= 0) {
    out.block_indices.clear();
    return true;
  }

  if (H5Lexists(group.get(), "block_offset", H5P_DEFAULT) > 0) {
    read_block_dims(logger, group.get(), "block_offset", out.block_offset);
  } else {
    out.block_offset = {0, 0, 0};
  }

  if (H5Lexists(group.get(), "block_indices", H5P_DEFAULT) <= 0) {
    RCLCPP_ERROR(logger, "HDF5: /compressed_data/block_indices が見つかりません");
    return false;
  }
  H5ObjectHandle dataset{H5Dopen2(group.get(), "block_indices", H5P_DEFAULT)};
  if (!dataset) {
    RCLCPP_ERROR(logger, "HDF5: block_indices のオープンに失敗しました");
    return false;
  }

  H5ObjectHandle dataspace{H5Dget_space(dataset.get())};
  if (!dataspace) {
    RCLCPP_ERROR(logger, "HDF5: block_indices の dataspace 取得に失敗しました");
    return false;
  }

  hssize_t element_count = H5Sget_simple_extent_npoints(dataspace.get());
  if (element_count <= 0) {
    RCLCPP_ERROR(logger, "HDF5: block_indices が空です");
    return false;
  }

  H5ObjectHandle dtype{H5Dget_type(dataset.get())};
  if (!dtype) {
    RCLCPP_ERROR(logger, "HDF5: block_indices の dtype 取得に失敗しました");
    return false;
  }

  const size_t expected_blocks = static_cast<size_t>(total_blocks);
  const size_t type_size = H5Tget_size(dtype.get());
  const H5T_class_t class_id = H5Tget_class(dtype.get());
  const bool is_integer = class_id == H5T_INTEGER;

  if (is_integer && static_cast<size_t>(element_count) == expected_blocks) {
    if (type_size == sizeof(uint8_t)) {
      std::vector<uint8_t> buffer(static_cast<size_t>(element_count));
      if (H5Dread(
          dataset.get(), H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()) < 0)
      {
        RCLCPP_ERROR(logger, "HDF5: block_indices(uint8) の読み込みに失敗しました");
        return false;
      }
      out.block_indices.resize(expected_blocks);
      out.sentinel_value = std::numeric_limits<uint8_t>::max();
      for (size_t i = 0; i < expected_blocks; ++i) {
        out.block_indices[i] = static_cast<uint32_t>(buffer[i]);
      }
      return true;
    }
    if (type_size == sizeof(uint16_t)) {
      std::vector<uint16_t> buffer(static_cast<size_t>(element_count));
      if (H5Dread(
          dataset.get(), H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()) < 0)
      {
        RCLCPP_ERROR(logger, "HDF5: block_indices(uint16) の読み込みに失敗しました");
        return false;
      }
      out.block_indices.resize(expected_blocks);
      out.sentinel_value = std::numeric_limits<uint16_t>::max();
      for (size_t i = 0; i < expected_blocks; ++i) {
        out.block_indices[i] = static_cast<uint32_t>(buffer[i]);
      }
      return true;
    }
    if (type_size == sizeof(uint32_t)) {
      std::vector<uint32_t> buffer(static_cast<size_t>(element_count));
      if (H5Dread(
          dataset.get(), H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()) < 0)
      {
        RCLCPP_ERROR(logger, "HDF5: block_indices(uint32) の読み込みに失敗しました");
        return false;
      }
      out.block_indices = std::move(buffer);
      out.sentinel_value = std::numeric_limits<uint32_t>::max();
      return true;
    }
  }

  // Packed bitfield fallback
  if (type_size != sizeof(uint8_t)) {
    RCLCPP_ERROR(
      logger, "HDF5: block_indices の型がサポート外です (size=%zu, elements=%lld)",
      type_size, static_cast<long long>(element_count));
    return false;
  }

  if (out.block_index_bit_width == 0) {
    out.block_index_bit_width = 1;
  }

  const size_t packed_bytes = static_cast<size_t>(element_count);
  std::vector<uint8_t> packed(packed_bytes);
  if (H5Dread(
      dataset.get(), H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, packed.data()) < 0)
  {
    RCLCPP_ERROR(logger, "HDF5: block_indices のビット列読み込みに失敗しました");
    return false;
  }

  const size_t expected_bytes =
    (expected_blocks * static_cast<size_t>(out.block_index_bit_width) + 7) / 8;
  if (packed.size() != expected_bytes) {
    RCLCPP_ERROR(
      logger, "HDF5: block_indices のサイズが一致しません (期待=%zu, 実際=%zu)",
      expected_bytes, packed.size());
    return false;
  }

  out.block_indices.resize(expected_blocks, 0);
  const uint32_t mask = compute_bitmask(out.block_index_bit_width);
  out.sentinel_value = mask;

  for (size_t i = 0; i < expected_blocks; ++i) {
    uint32_t value = 0;
    for (uint8_t bit = 0; bit < out.block_index_bit_width; ++bit) {
      const size_t absolute_bit = i * static_cast<size_t>(out.block_index_bit_width) + bit;
      const size_t byte_index = absolute_bit >> 3U;
      const size_t bit_index = absolute_bit & 7U;
      const uint8_t byte = packed[byte_index];
      const uint8_t bit_value = static_cast<uint8_t>((byte >> bit_index) & 0x1U);
      value |= static_cast<uint32_t>(bit_value) << bit;
    }
    out.block_indices[i] = value;
  }

  return true;
}

bool load_compressed_map(
  const rclcpp::Logger & logger, const std::string & filepath, CompressedMap & out)
{
  H5ObjectHandle file{H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)};
  if (!file) {
    RCLCPP_ERROR(logger, "HDF5: ファイルを開けませんでした: %s", filepath.c_str());
    return false;
  }

  if (!read_metadata(logger, file.get(), out)) {
    return false;
  }
  if (!read_compression_params(logger, file.get(), out)) {
    return false;
  }
  if (!read_dictionary(logger, file.get(), out)) {
    return false;
  }
  if (!read_compressed_data(logger, file.get(), out)) {
    return false;
  }

  return true;
}

std::vector<std::vector<std::array<int16_t, 3>>> build_pattern_voxels(
  const CompressedMap & map, std::vector<size_t> & counts)
{
  const uint32_t bits_per_pattern = map.pattern_length;
  const uint32_t block_size = std::max<uint32_t>(1, map.block_size);
  const uint32_t voxels_per_block = block_size * block_size * block_size;
  const uint32_t usable_bits = std::min(bits_per_pattern, voxels_per_block);
  const size_t bytes_per_pattern = (static_cast<size_t>(usable_bits) + 7) / 8;

  if (usable_bits == 0 || bytes_per_pattern == 0 || map.dictionary_patterns.empty()) {
    counts.clear();
    return {};
  }

  const size_t total_bytes = map.dictionary_patterns.size();
  if (total_bytes % bytes_per_pattern != 0) {
    // 端数は切り捨てる
  }
  const size_t pattern_count = total_bytes / bytes_per_pattern;

  std::vector<std::vector<std::array<int16_t, 3>>> patterns(pattern_count);
  counts.resize(pattern_count, 0);

  const uint32_t block_area = block_size * block_size;
  for (size_t pid = 0; pid < pattern_count; ++pid) {
    const uint8_t * base = map.dictionary_patterns.data() + pid * bytes_per_pattern;
    auto & bucket = patterns[pid];
    bucket.reserve(block_size * block_size);  // rough estimate

    for (uint32_t bit = 0; bit < usable_bits; ++bit) {
      const uint32_t byte_index = bit >> 3U;
      const uint32_t bit_index = bit & 7U;
      const uint8_t byte = base[byte_index];
      if (((byte >> bit_index) & 0x1U) == 0U) {
        continue;
      }
      const uint32_t linear = bit;
      const uint32_t z = linear / block_area;
      const uint32_t rem = linear % block_area;
      const uint32_t y = rem / block_size;
      const uint32_t x = rem % block_size;
      bucket.push_back({static_cast<int16_t>(x), static_cast<int16_t>(y), static_cast<int16_t>(z)});
    }
    counts[pid] = bucket.size();
  }

  return patterns;
}

std::vector<geometry_msgs::msg::Point> reconstruct_points(const CompressedMap & map)
{
  std::vector<size_t> pattern_counts;
  auto pattern_voxels = build_pattern_voxels(map, pattern_counts);
  if (pattern_voxels.empty()) {
    return {};
  }

  const size_t dictionary_size = pattern_voxels.size();
  const double voxel_size = static_cast<double>(map.voxel_size);
  const double block_span = voxel_size * static_cast<double>(map.block_size);
  const auto dims = map.block_dims;
  const auto offset = map.block_offset;

  const auto is_valid_pattern = [&](uint32_t idx) -> bool {
    return idx < dictionary_size && idx < pattern_counts.size();
  };
  const bool sentinel_is_dummy = map.sentinel_value >= dictionary_size;

  size_t total_points = 0;
  for (uint32_t index : map.block_indices) {
    if (index == map.sentinel_value && sentinel_is_dummy) {
      continue;
    }
    if (!is_valid_pattern(index)) {
      continue;
    }
    total_points += pattern_counts[index];
  }

  std::vector<geometry_msgs::msg::Point> points;
  points.reserve(total_points);

  size_t cursor = 0;
  for (int32_t bz = 0; bz < dims[2]; ++bz) {
    const double block_origin_z =
      map.grid_origin[2] + static_cast<double>(bz + offset[2]) * block_span;
    for (int32_t by = 0; by < dims[1]; ++by) {
      const double block_origin_y =
        map.grid_origin[1] + static_cast<double>(by + offset[1]) * block_span;
      for (int32_t bx = 0; bx < dims[0]; ++bx) {
        if (cursor >= map.block_indices.size()) {
          break;
        }
        const uint32_t pattern_id = map.block_indices[cursor++];
        if (pattern_id == map.sentinel_value) {
          continue;
        }
        if (!is_valid_pattern(pattern_id)) {
          continue;
        }
        const auto & voxels = pattern_voxels[pattern_id];
        if (voxels.empty()) {
          continue;
        }
        const double block_origin_x =
          map.grid_origin[0] + static_cast<double>(bx + offset[0]) * block_span;
        for (const auto & rel : voxels) {
          geometry_msgs::msg::Point pt;
          pt.x = block_origin_x + (static_cast<double>(rel[0]) + 0.5) * voxel_size;
          pt.y = block_origin_y + (static_cast<double>(rel[1]) + 0.5) * voxel_size;
          pt.z = block_origin_z + (static_cast<double>(rel[2]) + 0.5) * voxel_size;
          points.push_back(pt);
        }
      }
    }
  }

  return points;
}

}  // namespace

class VQServerNode : public rclcpp::Node
{
public:
  VQServerNode()
  : rclcpp::Node("vq_server")
  {
    const std::string default_frame = "map";
    const std::string default_topic = "compressed_map_markers";
    std::string default_map_path;
    try {
      const auto share_dir = ament_index_cpp::get_package_share_directory("vq_server");
      default_map_path = share_dir + "/maps/tsudanuma_voxelsize_05_compressed_map.h5";
    } catch (const std::exception & ex) {
      RCLCPP_WARN(
        get_logger(),
        "デフォルトマップのパス取得に失敗しました: %s",
        ex.what());
    }
    declare_parameter<std::string>("map_file", default_map_path);
    declare_parameter<std::string>("frame_id", default_frame);
    declare_parameter<std::string>("marker_topic", default_topic);
    declare_parameter<std::vector<double>>(
      "marker_color_rgba", std::vector<double>{0.1, 0.8, 0.4, 1.0});

    map_filepath_ = get_parameter("map_file").as_string();
    frame_id_param_ = get_parameter("frame_id").as_string();
    marker_topic_ = get_parameter("marker_topic").as_string();
    const auto marker_color_param = get_parameter("marker_color_rgba").as_double_array();
    std::array<float, 4> marker_color{0.1F, 0.8F, 0.4F, 1.0F};
    if (marker_color_param.size() >= 3) {
      for (size_t i = 0; i < 3; ++i) {
        marker_color[i] =
          static_cast<float>(std::clamp(marker_color_param[i], 0.0, 1.0));
      }
      if (marker_color_param.size() >= 4) {
        marker_color[3] =
          static_cast<float>(std::clamp(marker_color_param[3], 0.0, 1.0));
      }
    } else if (!marker_color_param.empty()) {
      RCLCPP_WARN(
        get_logger(), "marker_color_rgba パラメータには最低3つの値 (RGB) が必要です");
    }

    if (marker_topic_.empty()) {
      marker_topic_ = default_topic;
    }
    if (frame_id_param_.empty()) {
      frame_id_param_ = default_frame;
    }

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, qos);

    if (map_filepath_.empty()) {
      RCLCPP_ERROR(get_logger(), "map_file パラメータが設定されていません");
      return;
    }

    CompressedMap map;
    if (!load_compressed_map(get_logger(), map_filepath_, map)) {
      RCLCPP_ERROR(get_logger(), "HDF5ファイルの読み込みに失敗しました: %s", map_filepath_.c_str());
      return;
    }

    // フレームIDはパラメータ優先
    map.frame_id = frame_id_param_;

    auto points = reconstruct_points(map);
    RCLCPP_INFO(
      get_logger(), "ボクセル数: %zu (ブロック: %zu)", points.size(), map.block_indices.size());

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = map.frame_id;
    marker.header.stamp = now();
    marker.ns = "vq_compressed_map";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = map.voxel_size;
    marker.scale.y = map.voxel_size;
    marker.scale.z = map.voxel_size;
    marker.color.r = marker_color[0];
    marker.color.g = marker_color[1];
    marker.color.b = marker_color[2];
    marker.color.a = marker_color[3];
    marker.points = std::move(points);

    visualization_msgs::msg::MarkerArray array_msg;
    array_msg.markers.emplace_back(std::move(marker));

    publisher_->publish(array_msg);
    RCLCPP_INFO(get_logger(), "MarkerArray を1回だけ配信しました (topic: %s)", marker_topic_.c_str());
  }

private:
  std::string map_filepath_;
  std::string frame_id_param_;
  std::string marker_topic_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VQServerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
