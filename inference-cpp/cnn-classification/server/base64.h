//
// Code source: https://renenyffenegger.ch/notes/development/Base64/Encoding-and-decoding-base-64-with-cpp

//

#ifndef PROJECT_BASE64_H
#define PROJECT_BASE64_H
#include <string>

std::string base64_encode(unsigned char const* , unsigned int len);
std::string base64_decode(std::string const& s);

#endif //PROJECT_BASE64_H
