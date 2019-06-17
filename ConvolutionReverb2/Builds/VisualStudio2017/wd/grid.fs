#version 330 core

in vec2 uv;
uniform sampler2D tex;

void main()
{
	gl_FragColor = texture(tex, uv);
} 