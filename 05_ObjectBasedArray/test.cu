#include <iostream>
using namespace std;

int ReturnSIX(){
    return 6;
}

int ReturnTWO(){
    return 8;
}

class Point
{
private:
    int x, y;
 
public:
    // Parameterized Constructor
    Point(int x1, int y1)
    {
        x = x1;
        y = y1;
    }
 
    int getX()
    {
        return x;
    }
    int getY()
    {
        return y;
    }

    

    int AddXY()
    {
        int a = ReturnSIX();
        int b = ReturnTWO();
        return a + b;
    }

    int ReturnTWO(){
        return 2;
    }

};
 
int main()
{
    // Constructor called
    Point p1(10, 15);
 
    // Access values assigned by constructor
    cout << "p1.x = " << p1.getX() << ", p1.y = " << p1.getY() << "\n";
    cout << p1.AddXY() << "\n";
    return 0;
}