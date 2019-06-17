#pragma once

#include <string>
#include <map>
#include <functional>


class Shader
{
	
public:

	Shader();
	~Shader();

	bool attach(const std::string& source, int type);
	bool link();
	void use();
	bool validate();
	bool isValid() const { return programID > 0 && valid; }

	int getUniformLocation(const std::string& name);

	static void deactivateCurrentShader();

	unsigned int getProgramID() { return programID; }

	void with(const std::function<void()>&);

private:
	unsigned int programID = 0;
	std::map<std::string, int> uniformLocations;
	bool valid = false;


};
