#version 330 core

in vec2 uv;
uniform sampler2D tex;
uniform sampler1D lut;

void main()
{
	float value = 1.0 / (1.0 + exp(-10*texture(tex, uv).r));
	gl_FragColor = texture(lut, value);
} 