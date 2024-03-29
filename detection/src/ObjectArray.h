// Generated by gencpp from file detection/ObjectArray.msg
// DO NOT EDIT!


#ifndef DETECTION_MESSAGE_OBJECTARRAY_H
#define DETECTION_MESSAGE_OBJECTARRAY_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include "Object.h"

namespace detection
{
template <class ContainerAllocator>
struct ObjectArray_
{
  typedef ObjectArray_<ContainerAllocator> Type;

  ObjectArray_()
    : header()
    , objects()  {
    }
  ObjectArray_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , objects(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector< ::detection::Object_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::detection::Object_<ContainerAllocator> >::other >  _objects_type;
  _objects_type objects;





  typedef boost::shared_ptr< ::detection::ObjectArray_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::detection::ObjectArray_<ContainerAllocator> const> ConstPtr;

}; // struct ObjectArray_

typedef ::detection::ObjectArray_<std::allocator<void> > ObjectArray;

typedef boost::shared_ptr< ::detection::ObjectArray > ObjectArrayPtr;
typedef boost::shared_ptr< ::detection::ObjectArray const> ObjectArrayConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::detection::ObjectArray_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::detection::ObjectArray_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace detection

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'detection': ['/home/jetson/zoo_ws/src/animal_ir_detection/detection/msg'], 'sensor_msgs': ['/opt/ros/melodic/share/sensor_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/melodic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/melodic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::detection::ObjectArray_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::detection::ObjectArray_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::detection::ObjectArray_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::detection::ObjectArray_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::detection::ObjectArray_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::detection::ObjectArray_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::detection::ObjectArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "7c81ab9bad8cbdd82f821f9918564d8f";
  }

  static const char* value(const ::detection::ObjectArray_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x7c81ab9bad8cbdd8ULL;
  static const uint64_t static_value2 = 0x2f821f9918564d8fULL;
};

template<class ContainerAllocator>
struct DataType< ::detection::ObjectArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "detection/ObjectArray";
  }

  static const char* value(const ::detection::ObjectArray_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::detection::ObjectArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n"
"Object[] objects\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: detection/Object\n"
"sensor_msgs/RegionOfInterest bbox\n"
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

  static const char* value(const ::detection::ObjectArray_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::detection::ObjectArray_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.objects);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ObjectArray_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::detection::ObjectArray_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::detection::ObjectArray_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "objects[]" << std::endl;
    for (size_t i = 0; i < v.objects.size(); ++i)
    {
      s << indent << "  objects[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::detection::Object_<ContainerAllocator> >::stream(s, indent + "    ", v.objects[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // DETECTION_MESSAGE_OBJECTARRAY_H
