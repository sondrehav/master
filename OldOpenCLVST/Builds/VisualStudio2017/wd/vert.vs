#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uvcoord;

out vec2 uv;

void main()
{
	uv = uvcoord;
    gl_Position = vec4(position * 2, 1.0);
}