#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#define __STR_HELPER(X) #X
#define STR(X) __STR_HELPER(X)

#define DISABLE_DEFAULT_CONSTRUCTOR(class_name)             \
    class_name(const class_name &) = delete;                \
    class_name(class_name &&) = delete;                     \
    class_name &operator=(const class_name &) = delete;     \
    class_name &operator=(class_name &&) = delete;

#define ENABLE_DEFAULT_CONSTRUCTOR(class_name)              \
    class_name(const class_name &) = default;               \
    class_name(class_name &&) = default;                    \
    class_name &operator=(const class_name &) = default;    \
    class_name &operator=(class_name &&) = default;

#define CLASS_TEMPLATE_SPECIFICATION_HELPER(class_name, ...) \
    template class class_name<__VA_ARGS__>;

#define PI 3.141592653589793238462643383279502884L
#endif // GLOBAL_HPP
