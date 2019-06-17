#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uvcoord;

out vec2 uv;

uniform mat4 model;

void main()
{
	uv = uvcoord;
    gl_Position = model * vec4(position, 1.0);
}