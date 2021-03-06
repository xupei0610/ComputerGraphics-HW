"#version 330 core\n"
"out vec4 color;"
""
"in vec2 tex_coords;"
""
"uniform sampler2D diffuse;"
"uniform vec3 ambient;"
""
"void main()"
"{"
"   color = vec4(texture(diffuse, tex_coords).rgb * ambient, 1.0);"
"}";