# version

```cpp
#include "flop/version.hpp"
// namespace flop
```

```cpp
std::string_view version() noexcept;
```

The four-component label from `CMakeLists.txt`, compiled into the library.
A consumer asks the binary rather than trusting the headers it found; the
test runner prints it after every run so a log from a stale binary is
recognisable.

## Example

```cpp
#include <iostream>
#include "flop/version.hpp"

int main() { std::cout << "FLOP " << flop::version() << '\n'; }
```
