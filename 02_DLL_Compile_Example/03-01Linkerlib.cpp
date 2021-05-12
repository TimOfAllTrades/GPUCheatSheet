#include <iostream>
using namespace std;

//Compile code: cl /LD /Fe[Dlloutputfile].dll [DLLfilename].cpp
//Compile from visual studio developer prompt since that is where cl is
//This will output a dll, an o file and a lib file.

__declspec(dllexport) void hell()
{
    cout << "hell" << endl;
}