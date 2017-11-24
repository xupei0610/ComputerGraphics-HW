"#version 410 core\n"
"layout (location = 0) in vec4 v;"
"uniform mat4 proj;"
""
"out vec2 t_coord;"
""
"void main()"
"{"
"   gl_Position = proj * vec4(v.xy, 0.0, 1.0);"
"   t_coord = v.zw;"
"}"