// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Date:     2018/10/15

#include "./Imagine/Features.h"
#include <Imagine/LinAlg.h>

using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color, 2> I1, Image<Color, 2> I2,
              vector<Match> &matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0, 0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(), 0));
    cout << " Im2: " << feats2.size() << flush << endl;

    const double MAX_DISTANCE = 100.0 * 100.0;
    for (size_t i = 0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1 = feats1[i];
        for (size_t j = 0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if (d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

Matrix<float> computeA(vector<Match> points) {
    Matrix<float> A(9, 9);
    for (int j = 0; j < points.size(); j++) {
        A(0, j) = points[j].x1 * points[j].x2;
        A(1, j) = points[j].x2 * points[j].y1;
        A(2, j) = points[j].x2;
        A(3, j) = points[j].y2 * points[j].x1;
        A(4, j) = points[j].y2 * points[j].y1;
        A(5, j) = points[j].y2;
        A(6, j) = points[j].x1;
        A(7, j) = points[j].y1;
        A(8, j) = 1;
        A(j, 8) = 0;
    }
    return A;
}

float computeEpipolarDistance(Match m, Matrix<float> F) {
    Vector<float> u(3);
    u[0] = m.x1;
    u[1] = m.y1;
    u[2] = 1;
    Vector<float> v(3);
    v[0] = m.x2;
    v[1] = m.y2;
    v[2] = 1;
    v = F * v;
    v = v / (sqrt(pow(v[0], 2.) + pow(v[1], 2.)));
    return abs(u[0]*v[0]+u[1]*v[1]+v[2]);
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float, 3, 3> computeF(vector<Match> &matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    float Niter = 100000; // Adjusted dynamically
    FMatrix<float, 3, 3> bestF;
    vector<int> bestInliers;
    int nbInliers;
    int n_samples = 8;

    int counter = 0;
    Matrix<float> A;
    Matrix<float> U, V;
    Vector<float> f;
    Vector<float> S;
    Matrix<float> F(3, 3);
    vector<int> inliers;
    cout << "Computing Fundamental Matrix, please wait ..." << endl;
    while (counter < Niter) {
        std::random_shuffle(matches.begin(), matches.end());
        vector<Match> subPoints(matches.begin(), matches.begin() + n_samples);

        // Normalize the eight points chosen for computing the model
        for (size_t i = 0; i < subPoints.size(); i++) {
            subPoints[i].x1 *= 10e-3;
            subPoints[i].x2 *= 10e-3;
            subPoints[i].y1 *= 10e-3;
            subPoints[i].y2 *= 10e-3;
        }
        // Compute A matrix and adding a row of zeros to have a square matrix
        A = computeA(subPoints);

        // Compute SVD of A
        svd(A, U, S, V);
        f = U.getCol(8);
        F(0, 0) = f[0];
        F(0, 1) = f[1];
        F(0, 2) = f[2];
        F(1, 0) = f[3];
        F(1, 1) = f[4];
        F(1, 2) = f[5];
        F(2, 0) = f[6];
        F(2, 1) = f[7];
        F(2, 2) = f[8];

        // Compute SVD of F to enforce constraint of rank
        svd(F, U, S, V);
        S[2] = 0;
        F = U * Diagonal(S) * transpose(V);

        // Normalizing F
        Vector<float> N(3);
        N[0] = 10e-3;
        N[1] = 10e-3;
        N[2] = 1.0;
        F = Diagonal(N) * F * Diagonal(N);
        for (int i = 0; i < matches.size(); i++) {
            if (computeEpipolarDistance(matches[i], F) <= distMax) {
                inliers.push_back(i);
            }
        }

        if (inliers.size() > bestInliers.size()) {
            bestF(0, 0) = F(0, 0);
            bestF(0, 1) = F(0, 1);
            bestF(0, 2) = F(0, 2);
            bestF(1, 0) = F(1, 0);
            bestF(1, 1) = F(1, 1);
            bestF(1, 2) = F(1, 2);
            bestF(2, 0) = F(2, 0);
            bestF(2, 1) = F(2, 1);
            bestF(2, 2) = F(2, 2);
            bestInliers = inliers;
            nbInliers = (int) bestInliers.size();
            if (log(1. - pow((float) nbInliers / (float) matches.size(), n_samples)) != 0) {
                Niter = ceil((float) log(BETA) / (float) log(1. - pow((float) nbInliers / (float) matches.size(), n_samples)));
            }
        }
        inliers.clear();
        counter++;
    }
    cout << "Final Niter: " << Niter << endl;

    // Updating matches with inliers only
    vector<Match> all = matches;
    matches.clear();
    for (size_t i = 0; i < bestInliers.size(); i++) {
        matches.push_back(all[bestInliers[i]]);
    }

    cout << "Best F-Matrix has been computed successfully !" << endl;
    return bestF;
}


// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float, 3, 3> &F) {
    cout << "Left click to select a point and draw its epipolar line in the other image." << endl;
    cout << "Quit by right clicking on the window." << endl;
    int w1 = I1.width();
    int w2 = I2.width();

    while (true) {
        int x, y;
        if (getMouse(x, y) == 3) {
            break;
        }

        if (getMouse(x, y) == 1) {
            IntPoint2 p(x, y);
            cout << "Point clicked:" << p << endl;
            drawCircle(p, 4, RED, 2);
            bool epipolarImage = x < w1;
            FVector<float, 3> v;
            v[1] = y;
            v[2] = 1;
            IntPoint2 leftPoint, rightPoint;
            if (epipolarImage) {
                cout << "Epipolar view is on the right image" << endl;
                v[0] = x;
                v = transpose(F) * v;
                v /= v[2];
                leftPoint[0] = w1;
                leftPoint[1] = (int) (-v[2] / v[1]);
                rightPoint[0] = w1+w2;
                rightPoint[1] = (int) (-(v[2] + v[0] * w1) / v[1]);
            } else {
                cout << "Epipolar view is on the left image" << endl;
                v[0] = x-w1;
                v = F * v;
                v /= v[2];
                leftPoint[0] = 0;
                leftPoint[1] = -v[2]/v[1];
                rightPoint[0] = w1;
                rightPoint[1] = (int) (-(v[2] + v[0] * w1) / v[1]);
            }

            cout << "v=" << v << endl;

            cout << "The line goes from " << rightPoint << " to  " << leftPoint << endl;
            drawLine(leftPoint, rightPoint, YELLOW, 2);
        }
    }
}

int main(int argc, char *argv[]) {
    srand((unsigned int) time(0));

    const char *s1 = argc > 1 ? argv[1] : srcPath("im1.jpg");
    const char *s2 = argc > 2 ? argv[2] : srcPath("im2.jpg");

    // Load and display images
    Image<Color, 2> I1, I2;
    if (!load(I1, s1) ||
        !load(I2, s2)) {
        cerr << "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2 * w, I1.height());
    display(I1, 0, 0);
    display(I2, w, 0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    cout << "Number of matches: " << matches.size() << endl;
    cout << "Click to compute the fundamental matrix F" << endl;
    click();

    FMatrix<float, 3, 3> F = computeF(matches);
    cout << "F=" << F << endl;

    // Redisplay with matches
    display(I1, 0, 0);
    display(I2, w, 0);
    for (size_t i = 0; i < matches.size(); i++) {
        Color c(rand() % 256, rand() % 256, rand() % 256);
        fillCircle(matches[i].x1 + 0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2 + w, matches[i].y2, 2, c);
    }
    cout << "Click to redisplay images without matches to draw epipolar lines" << endl;
    click();

    // Redisplay without SIFT points
    display(I1, 0, 0);
    display(I2, w, 0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
