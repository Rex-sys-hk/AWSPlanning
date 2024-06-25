#include "collisiondetection.h"

using namespace HybridAStar;

CollisionDetection::CollisionDetection() {
  this->grid = nullptr;
  Lookup::collisionLookup(collisionLookup);
}

bool CollisionDetection::configurationTest(float x, float y, float t) const {
  int X = (int)x;
  int Y = (int)y;
  int iX = (int)((x - (long)x) * Constants::positionResolution);
  iX = iX > 0 ? iX : 0;
  int iY = (int)((y - (long)y) * Constants::positionResolution);
  iY = iY > 0 ? iY : 0;
  int iT = (int)((Helper::normalizeHeadingRad(t)+M_PI) / Constants::deltaHeadingRad);
  int idx = iY * Constants::positionResolution * Constants::headings + iX * Constants::headings + iT;
  int cX;
  int cY;
  for (int i = 0; i < collisionLookup[idx].length; ++i) {
    cX = (X + collisionLookup[idx].pos[i].x);
    cY = (Y + collisionLookup[idx].pos[i].y);
    // make sure the configuration coordinates are actually on the grid
    if (cX >= 0 && (unsigned int)cX < grid->info.width 
     && cY >= 0 && (unsigned int)cY < grid->info.height) {
      if (grid->data[cY * grid->info.width + cX]) {
        return false;
      }
    }
  }

  return true;
}

bool CollisionDetection::configurationOnline(float x, float y, float t) const {
    int X = (int)x;
    int Y = (int)y;
    float length = Constants::length;
    float width = Constants::width;
    int l_sample = ceil(length / Constants::cellSize);
    float delta_l = length / (l_sample-1);
    int w_sample = ceil(width / Constants::cellSize);
    float delta_w = width / (w_sample-1);
    for (int i = 0; i < l_sample; ++i) {
        for (int j = 0; j < w_sample; ++j) {
            float x_ = x + 
                      ((-length/2 + i * delta_l) * cos(t) 
                      - (-width/2 + j * delta_w) * sin(t))
                      ;
            float y_ = y +
                      ((-length/2 + i * delta_l) * sin(t) 
                      + (-width/2 + j * delta_w) * cos(t))
                      ;
            int cX = (int)(x_);
            int cY = (int)(y_);
            if (cX >= 0 && (unsigned int)cX < grid->info.width 
             && cY >= 0 && (unsigned int)cY < grid->info.height) {
                if (grid->data[cY * grid->info.width + cX]) {
                    return false;
                }
            }
        }
    }
    return true;
}