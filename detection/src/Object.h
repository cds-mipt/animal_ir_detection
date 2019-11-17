// Generated by gencpp from file detection/Object.msg
// DO NOT EDIT!


#ifndef DETECTION_MESSAGE_OBJECT_H
#define DETECTION_MESSAGE_OBJECT_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <sensor_msgs/RegionOfInterest.h>

namespace detection
{
template <class ContainerAllocator>
struct Object_
{
  typedef Object_<ContainerAllocator> Type;

  Object_()
    : bbox()
    , label(0)  {
    }
  Object_(const ContainerAllocator& _alloc)
    : bbox(_alloc)
    , label(0)  {
  (void)_alloc;
    }



   typedef  ::sensor_msgs::RegionOfInterest_<ContainerAllocator>  _bbox_type;
  _bbox_type bbox;

   typedef uint8_t _label_type;
  _label_type label;





  typedef boost::shared_ptr< ::detection::Object_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::detection::Object_<ContainerAllocator> const> ConstPtr;

}; // struct Object_

typedef ::detection::Object_<std::allocator<void> > Object;

typedef boost::shared_ptr< ::detection::Object > ObjectPtr;
typedef boost::shared_ptr< ::detection::Object const> ObjectConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::detection::Object_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::detection::Object_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace detection

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'detection': ['/home/jetson/zoo_ws/src/animal_ir_detection/detection/msg'], 'sensor_msgs': ['/opt/ros/melodic/share/sensor_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/melodic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/melodic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::detection::Object_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::detection::Object_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::detection::Object_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::detection::Object_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::detection::Object_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::detection::Object_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::detection::Object_<ContainerAllocator> >
{
  static const char* value()
  {
    return "27802822030f9624a9f7680aba9adb7b";
  }

  static const char* value(const ::detection::Object_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x27802822030f9624ULL;
  static const uint64_t static_value2 = 0xa9f7680aba9adb7bULL;
};

template<class ContainerAllocator>
struct DataType< ::detection::Object_<ContainerAllocator> >
{
  static const char* value()
  {
    return "detection/Object";
  }

  static const char* value(const ::detection::Object_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::detection::Object_<ContainerAllocator> >
{
  static const char* value()
  {
    return "sensor_msgs/RegionOfInterest bbox\n"
"uint8 label\n"
"================================================================================\n"
"MSG: sensor_msgs/RegionOfInterest\n"
"# This message is used to specify a region of interest within an image.\n"
"#\n"
"# When used to specify the ROI setting of the camera when the image was\n"
"# taken, the height and width fields should either match the height and\n"
"# width fields for the associated image; or height = width = 0\n"
"# indicates that the full resolution image was captured.\n"
"\n"
"uint32 x_offset  # Leftmost pixel of the ROI\n"
"                 # (0 if the ROI includes the left edge of the image)\n"
"uint32 y_offset  # Topmost pixel of the ROI\n"
"                 # (0 if the ROI includes the top edge of the image)\n"
"uint32 height    # Height of ROI\n"
"uint32 width     # Width of ROI\n"
"\n"
"# True if a distinct rectified ROI should be calculated from the \"raw\"\n"
"# ROI in this message. Typically this should be False if the full image\n"
"# is captured (ROI not used), and True if a subwindow is captured (ROI\n"
"# used).\n"
"bool do_rectify\n"
;
  }

  static const char* value(const ::detection::Object_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::detection::Object_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.bbox);
      stream.next(m.label);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Object_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::detection::Object_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::detection::Object_<ContainerAllocator>& v)
  {
    s << indent << "bbox: ";
    s << std::endl;
    Printer< ::sensor_msgs::RegionOfInterest_<ContainerAllocator> >::stream(s, indent + "  ", v.bbox);
    s << indent << "label: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.label);
  }
};

} // namespace message_operations
} // namespace ros

#endif // DETECTION_MESSAGE_OBJECT_H
