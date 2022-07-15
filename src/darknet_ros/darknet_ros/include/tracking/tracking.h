#ifndef TRACKING_H
#define TRACKING_H

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
 
using namespace cv;
using namespace std;

class TrackObject {
private:
    uint16_t _id;
    uint64_t _live_time;
    Ptr<Tracker> tracker;
public
    TrackObject(uint16_t id, uint64_t live_time, cv::Mat frame, const cv::Rect &roi): 
        _id(id), _live_time(live_time) {
        tracker = TrackerKCF::create();
        tracker.init(frame, roi);
    }

    void setLiveTime(uint64_t live_time) {
        _live_time = live_time;
    }

    uint64_t getLiveTime() const {
        return _live_time;
    }

};

class Tracking {
private:
    vector<shared_ptr
public:
    Tracking() {
        
    }
};

#endif
