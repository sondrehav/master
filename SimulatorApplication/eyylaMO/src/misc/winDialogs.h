#pragma once

#include <string>
#include <functional>

void openDialog(std::function<void(bool aborted, const std::string& file)> cb);
void saveDialog(std::function<void(bool aborted, const std::string& file)> cb);
void errorMsg(const std::string& msg);