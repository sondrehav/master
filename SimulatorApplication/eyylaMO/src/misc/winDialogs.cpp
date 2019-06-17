#include "winDialogs.h"
#include <windows.h>
#include <commdlg.h>
#include <tchar.h>

void openDialog(std::function<void(bool aborted, const std::string& file)> cb)
{
	char filename[256];

	OPENFILENAME ofn;

	std::memset(&ofn, 0, sizeof(OPENFILENAME));
	std::memset(&filename, 0, sizeof(char) * 256);

	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFilter = _T("Audio Files\0*.wav\0");
	ofn.lpstrFile = filename;
	ofn.nMaxFile = 256;
	ofn.lpstrTitle = _T("Select a File, yo!");
	ofn.Flags = OFN_FILEMUSTEXIST;

	if (GetOpenFileName(&ofn))
	{
		cb(false, filename);
	}
	else
	{
		cb(true, "");
	}
}


void saveDialog(std::function<void(bool aborted, const std::string& file)> cb)
{
	char filename[256];

	OPENFILENAME ofn;

	std::memset(&ofn, 0, sizeof(OPENFILENAME));
	std::memset(&filename, 0, sizeof(char) * 256);

	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFilter = _T("Audio Files\0*.wav\0");
	ofn.lpstrFile = filename;
	ofn.nMaxFile = 256;
	ofn.lpstrTitle = _T("Select a File, yo!");
	ofn.Flags = OFN_FILEMUSTEXIST;

	if (GetSaveFileName(&ofn))
	{
		cb(false, filename);
	}
	else
	{
		cb(true, "");
	}
}

void errorMsg(const std::string& msg)
{
	char* title = "Error!";
	MessageBox(NULL, msg.c_str(), title, MB_OK);
}
