# WATO Topics Package

This package exports an interface library that contains header files for various subteams. Each header file contains the strings of topic names that they publish.

To subscribe to a particular subteam's topic, the appropriate header should be included and use the desired topic string.

To use the wato_topics package, do the following:


In the package.xml, add:

```
<build_depend>wato_topics</build_depend>
<run_depend>wato_topics</run_depend>
```

In the CMakeLists.txt file, add:

```
find_package(
     ... 
    wato_topics
)

catkin_package(
    ...
    wato_topics
)

target_link_libraries(
    ...
    wato_topics
 )
```
In the file that's dependent on the topic name, add:
```
#include <wato_topics/perception_topics.h>
...
string test = perception_topics::STOP_LINE_TOPIC;
...
```
