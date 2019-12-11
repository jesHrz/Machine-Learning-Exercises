#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#define inf 0x3f3f3f3f
#define cases(t) for (int cas = 1; cas <= int(t); ++cas)
typedef long long ll;
typedef double db;
using namespace std;

#ifdef NO_ONLINE_JUDGE
#define LOG(args...) do { cout << #args << " -> "; err(args); } while (0)
void err() { cout << endl; }
template<typename T, typename... Args> void err(T a, Args... args) { cout << a << ' '; err(args...); }
#else
#define LOG(...)
#endif

const db eps = 1e-6;

db x[80][3];
db y[80];
db w[3], ww[3];

db sig(db x) {
    return 1.0 / (1.0 + exp(-x));
}

db cal() {
    db ret = 0;
    for (int i = 0; i < 80; ++i) {
        db h = 0;
        for (int k = 0; k < 3; ++k)	h += w[k] * x[i][k];
        h = sig(h);
        ret += -y[i] * log(h) - (1 - y[i]) * log(1 - h);
    }
    return ret / 80;
}

vector<db> fit(db alpha) {
    for (int i = 0; i < 3; ++i)	w[i] = ww[i] = 0;
    db val = 0, last;
    vector<db> cost;
    while (1) {
        for (int j = 0; j < 3; ++j) {
            db J = 0;
            for (int i = 0; i < 80; ++i) {
                db h = 0;
                for (int k = 0; k < 3; ++k) h += ww[k] * x[i][k];
                J += (sig(h) - y[i]) * x[i][j];
            }
            w[j] -= alpha * J / 80;
        }
        for (int i = 0; i < 3; ++i)	ww[i] = w[i];
        last = val;
        val = cal();
        cost.push_back(val);
        LOG(val - last);
        if (fabs(val - last) <= eps)	break;
    }
    return cost;
}

int main() {
    ifstream fin("data/ex2x.dat");
    for (int i = 0; i < 80; ++i) {
        x[i][0] = 1;
        fin >> x[i][1] >> x[i][2];
    }
    fin.close();
    fin.open("data/ex2y.dat");
    for (int i = 0; i < 80; ++i) {
        fin >> y[i];
    }
    fin.close();
    // for (int i = 0; i < 80; ++i) {
    //     cout << x[i][0] << ' ' << x[i][1] << ' ' << x[i][2] << ' ' << y[i] << endl;
    // }
    db alpha = 0.01;
    vector<db> ret = fit(alpha);
    LOG(alpha, w[0], w[1], w[2]);
    // string name = "data/" + to_string(alpha) + ".txt";
    // ofstream fout(name.c_str());
    // fout << alpha << " " << w[0] << " " << w[1] << " " << w[2] << " " << cal() << endl;
    // for(auto v:ret)	fout << v << endl;
    return 0;
}
/*
alpha, w[0], w[1], w[2], fit(alpha) -> 0.0004 -0.00924545 0.0459596 -0.0229827 535
alpha, w[0], w[1], w[2], fit(alpha) -> 0.0005 -0.0110848 0.0466746 -0.0233549 514
alpha, w[0], w[1], w[2], fit(alpha) -> 0.0006 -0.43249 0.0491012 -0.0185842 17245
alpha, w[0], w[1], w[2], fit(alpha) -> 0.0007 -1.10884 0.0519389 -0.0103293 39377
alpha, w[0], w[1], w[2], fit(alpha) -> 0.0008 -1.66343 0.0543808 -0.00361771 53419
alpha, w[0], w[1], w[2], fit(alpha) -> 0.0009 -2.1325 0.0565278 0.00201761 62666
alpha, w[0], w[1], w[2], fit(alpha) -> 0.001 -2.53826 0.0584452 0.00686168 68898
alpha, w[0], w[1], w[2], fit(alpha) -> 0.0012 -3.21363 0.0617594 0.0148612 76044
alpha, w[0], w[1], w[2], fit(alpha) -> 0.0015 -4.10757 0.066715 0.0258728 80821
alpha, w[0], w[1], w[2], fit(alpha) -> 0.0018 -8.70063 0.0939862 0.0776859 183547
*/