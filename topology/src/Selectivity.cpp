#include "Selectivity.h"
#include <cereal/archives/binary.hpp>
// #include <cereal/archives/xml.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <membuf.h>
#include <iostream>

Selectivity::Selectivity(std::vector<JointDistributions> &distributions, uint32_t funcAttr, uint32_t cubeDim, uint32_t targetRes)
{
    mDistributions = distributions;
    mFuncAttr = funcAttr;
    mSize = mDistributions.size();
    mAttrs = mDistributions[mSize - 1].getAttr();
    mCubeDim = cubeDim;
    mTargetRes = targetRes;
}

std::vector<uint32_t> Selectivity::jointQuery(std::string attr1, std::string attr2)
{
    uint32_t targetRes = mTargetRes;
    uint32_t res = mDistributions[mSize - 1].get(attr1, attr2).resolution();

    std::vector<uint32_t> temp_hist(targetRes * targetRes, 0);

    Histogram tempHist = mDistributions[mSize - 1].get(attr1, attr2);

    temp_hist.resize(res * res);
    for (uint32_t ii = 0; ii < temp_hist.size(); ii++)
    {
        temp_hist[ii] = tempHist.data()[ii];
    }
    // ! Interpolation needed here
    return interpolateHist(temp_hist, res, targetRes);
}

std::vector<uint32_t> Selectivity::functionQuery(std::vector<std::string> &dims, std::vector<std::vector<float>> &ranges,
                                                 int32_t targetIndex, std::vector<uint32_t> selectedIndex)
{
    uint32_t targetRes = mTargetRes;
    // Right now it only knows how to query one attribute
    std::string query_attr = dims[0];

    std::string target_attr;

    if (targetIndex == -1)
    {
        targetIndex = mFuncAttr;
    }

    target_attr = mAttrs[targetIndex];

    std::vector<uint32_t> yHist;

    std::vector<std::pair<float, float>> query_range = mDistributions[mSize - 1].get(query_attr).ranges();

    float query_min = query_range[0].first;
    float query_max = query_range[0].second;

    float sel_query_min = ranges[0][0];
    float sel_query_max = ranges[0][1];

    if(sel_query_min<query_min)
        sel_query_min = query_min;
    if(sel_query_max>query_max)
        sel_query_max = query_max; 
    
    uint32_t res = mDistributions[mSize - 1].get(query_attr).resolution();

    uint32_t start_ind = (uint32_t)(res * (sel_query_min - query_min) / (query_max - query_min));
    uint32_t end_ind = (uint32_t)(res * (sel_query_max - query_min) / (query_max - query_min));

    // Query Attr is the same as target Attr
    if (query_attr == target_attr)
    {
        if (selectedIndex.size() == 0)
            yHist = jointQuery(dims); // THIS NEED TO BE CHANGED
        else
            yHist = jointQuery(selectedIndex, dims); // THIS NEED TO BE CHANGED

        for (uint32_t ind = 0; ind < yHist.size(); ind++)
        {
            if (ind < start_ind)
                yHist[ind] = 0;
            else if (ind > end_ind)
                yHist[ind] = 0;
        }
        // This should not care the original size of marginal distribution
        return interpolateHist(yHist, res, targetRes);
    }
    else
    {
        // bool rev = false;
        uint32_t src_ind;
        uint32_t target_ind;

        for (uint32_t ii = 0; ii < mAttrs.size(); ii++)
        {
            if (mAttrs[ii] == query_attr)
                src_ind = ii;
            if (mAttrs[ii] == target_attr)
                target_ind = ii;
        }

        if (selectedIndex.size() == 0)
            yHist = (src_ind < target_ind) ? jointQuery(query_attr, target_attr) : jointQuery(target_attr, query_attr);
        else
            yHist = (src_ind < target_ind) ? jointQuery(selectedIndex, query_attr, target_attr) : jointQuery(selectedIndex, target_attr, query_attr);

        std::vector<uint32_t> outHist(res, 0);

        for (uint32_t row = 0; row < res; row++)
        {
            for (uint32_t col = 0; col < res; col++)
            {

                if (src_ind < target_ind)
                {
                    if ((row >= start_ind) && (row <= end_ind))
                        outHist[col] += yHist[row * res + col];
                }
                else
                {
                    if ((col >= start_ind) && (col <= end_ind))
                        outHist[row] += yHist[row * res + col];
                }
            }
        }
        return interpolateHist(outHist, res, targetRes);
    }
}

std::vector<uint32_t> Selectivity::jointQuery(std::vector<std::string> &attrs, bool func,
                                              std::vector<std::string> dims, std::vector<std::vector<float>> ranges)
{
    uint32_t targetRes = mTargetRes;
    if (!func)
    {
        // ! Return 1D Histogram
        if (attrs.size() == 1)
        { // ! No Selection
            if (dims.size() == 0)
            {
                uint32_t res = mDistributions[0].get(attrs[0]).resolution();

                std::vector<uint32_t> temp_hist(res, 0);

                Histogram tempHist = mDistributions[mSize - 1].get(attrs[0]);

                for (uint32_t ii = 0; ii < temp_hist.size(); ii++)
                {
                    temp_hist[ii] = tempHist.data()[ii];
                }
                return interpolateHist(temp_hist, res, targetRes);
            }
            // ! 1D Selection
            else
            {
                //std::vector<std::string> mAttrs = mDistributions[mSize-1].getAttr();
                int32_t target_attr_index = -1;
                for (uint32_t ii = 0; ii < mAttrs.size(); ii++)
                {
                    if (mAttrs[ii] == attrs[0])
                        target_attr_index = ii;
                }
                // ! count will not affect anything here since this returns overall function Histogram
                return functionQuery(dims, ranges, target_attr_index);
            }
        }
        // Return 2D Histogram
        else
        {
            // No selection and max data cube stored <=2
            if ((dims.size() == 0) || (mCubeDim < 3))
                return jointQuery(attrs[0], attrs[1]);
            else
            {
                //std::vector<std::string> mAttrs = mDistributions[mSize-1].getAttr();

                int32_t src_attr_index = -1;
                int32_t src_attr_index2 = -1;
                int32_t selected_attr_index = -1;
                for (uint32_t ii = 0; ii < mAttrs.size(); ii++)
                {
                    if (mAttrs[ii] == attrs[0])
                        src_attr_index = ii;
                    if (mAttrs[ii] == attrs[1])
                        src_attr_index2 = ii;
                    if (mAttrs[ii] == dims[0])
                        selected_attr_index = ii;
                }

                uint32_t res = mDistributions[mSize - 1].get(dims[0]).resolution();
                std::vector<uint32_t> outHist(res * res, 0);

                std::vector<std::pair<float, float>> query_range = mDistributions[mSize - 1].get(dims[0]).ranges();

                float query_min = query_range[0].first;
                float query_max = query_range[0].second;

                float sel_query_min = ranges[0][0];
                float sel_query_max = ranges[0][1];

                if(sel_query_min<query_min)
                    sel_query_min = query_min;
                if(sel_query_max>query_max)
                    sel_query_max = query_max; 

                uint32_t start_ind = (uint32_t)(res * (sel_query_min - query_min) / (query_max - query_min));
                uint32_t end_ind = (uint32_t)(res * (sel_query_max - query_min) / (query_max - query_min));

                // Query for 2D or 3D depending on dims vs attrs

                if (src_attr_index == selected_attr_index)
                {
                    std::vector<uint32_t> src_hist = jointQuery(attrs[0], attrs[1]);
                    for (uint32_t row = 0; row < res; row++)
                    {
                        for (uint32_t col = 0; col < res; col++)
                        {
                            if ((row >= start_ind) && (row <= end_ind))
                                outHist[row * res + col] = src_hist[row * res + col];
                        }
                    }
                    return interpolateHist(outHist, res, targetRes);
                }
                else if (src_attr_index2 == selected_attr_index)
                {
                    std::vector<uint32_t> src_hist = jointQuery(attrs[0], attrs[1]);
                    for (uint32_t row = 0; row < res; row++)
                    {
                        for (uint32_t col = 0; col < res; col++)
                        {
                            if ((col >= start_ind) && (col <= end_ind))
                                outHist[row * res + col] = src_hist[row * res + col];
                        }
                    }
                    return interpolateHist(outHist, res, targetRes);
                }
                else
                {
                    // Get 3D Hist
                    if (selected_attr_index < src_attr_index)
                    {
                        Histogram tempHist = mDistributions[mSize - 1].get(dims[0], attrs[0], attrs[1]);
                        uint32_t res3D = tempHist.resolution();
                        outHist.resize(res3D * res3D, 0);
                        start_ind = (uint32_t)(res3D * (sel_query_min - query_min) / (query_max - query_min));
                        end_ind = (uint32_t)(res3D * (sel_query_max - query_min) / (query_max - query_min));
                        for (uint32_t row = 0; row < res3D; row++)
                        {
                            for (uint32_t col = 0; col < res3D; col++)
                            {
                                for (uint32_t lay = 0; lay < res3D; lay++)
                                {
                                    if ((row >= start_ind) && (row <= end_ind))
                                        outHist[col * res3D + lay] += tempHist.data()[row * res3D * res3D + col * res3D + lay];
                                }
                            }
                        }
                        std::vector<uint32_t> target_hist = jointQuery(attrs[0], attrs[1]);
                        return interpolateHist(outHist, target_hist, res3D, targetRes);
                    }
                    else if (selected_attr_index > src_attr_index2)
                    {
                        Histogram tempHist = mDistributions[mSize - 1].get(attrs[0], attrs[1], dims[0]);
                        uint32_t res3D = tempHist.resolution();
                        outHist.resize(res3D * res3D, 0);
                        start_ind = (uint32_t)(res3D * (sel_query_min - query_min) / (query_max - query_min));
                        end_ind = (uint32_t)(res3D * (sel_query_max - query_min) / (query_max - query_min));
                        for (uint32_t row = 0; row < res3D; row++)
                        {
                            for (uint32_t col = 0; col < res3D; col++)
                            {
                                for (uint32_t lay = 0; lay < res3D; lay++)
                                {
                                    if ((lay >= start_ind) && (lay <= end_ind))
                                        outHist[row * res3D + col] += tempHist.data()[row * res3D * res3D + col * res3D + lay];
                                }
                            }
                        }
                        std::vector<uint32_t> target_hist = jointQuery(attrs[0], attrs[1]);
                        return interpolateHist(outHist, target_hist, res3D, targetRes);
                    }
                    else
                    {
                        Histogram tempHist = mDistributions[mSize - 1].get(attrs[0], dims[0], attrs[1]);
                        uint32_t res3D = tempHist.resolution();
                        outHist.resize(res3D * res3D, 0);
                        start_ind = (uint32_t)(res3D * (sel_query_min - query_min) / (query_max - query_min));
                        end_ind = (uint32_t)(res3D * (sel_query_max - query_min) / (query_max - query_min));
                        for (uint32_t row = 0; row < res3D; row++)
                        {
                            for (uint32_t col = 0; col < res3D; col++)
                            {
                                for (uint32_t lay = 0; lay < res3D; lay++)
                                {
                                    if ((col >= start_ind) && (col <= end_ind))
                                        outHist[row * res3D + lay] += tempHist.data()[row * res3D * res3D + col * res3D + lay];
                                }
                            }
                        }
                        std::vector<uint32_t> target_hist = jointQuery(attrs[0], attrs[1]);
                        return interpolateHist(outHist, target_hist, res3D, targetRes);
                    }
                }
            }
        }
    }
    // ! return function distribution
    else
    { // ! No Selection on other dim
        if (dims.size() == 0)
        {
            std::string func_attr = mAttrs[mFuncAttr]; //mDistributions[mSize-1].getAttr()[mFuncAttr];
            std::vector<std::string> func_vec;
            func_vec.push_back(func_attr);
            return jointQuery(func_vec);
        }
        else
        { // ! When there is selection on other dim
            // ! count will not affect anything here since this returns overall function Histogram
            return functionQuery(dims, ranges);
        }
    }
}

std::vector<uint32_t> Selectivity::jointQuery(std::vector<uint32_t> &selectedIndex, std::vector<std::string> &attrs, bool func,
                                              std::vector<std::string> dims, std::vector<std::vector<float>> ranges)
{
    uint32_t targetRes = mTargetRes;
    if (!func)
    {
        // ! Return 1D Histogram
        if (attrs.size() == 1)
        {
            // ! No brush constraint
            if (dims.size() == 0)
            {
                uint32_t res = mDistributions[0].get(attrs[0]).resolution();

                std::vector<uint32_t> temp_hist(res, 0);

                for (uint32_t i = 0; i < (uint32_t)selectedIndex.size(); i++)
                {
                    Histogram tempHist = mDistributions[selectedIndex[i]].get(attrs[0]);
                    //  Changing this in someway would speed up the process
                    for (uint32_t ii = 0; ii < temp_hist.size(); ii++)
                    {
                        temp_hist[ii] += tempHist.data()[ii];
                    }
                }
                return interpolateHist(temp_hist, res, targetRes);
            }
            // ! Wit Brush
            else
            {
                int32_t query_attr_index = -1;
                for (uint32_t ii = 0; ii < mAttrs.size(); ii++)
                {
                    if (mAttrs[ii] == attrs[0])
                        query_attr_index = ii;
                }
                return functionQuery(dims, ranges, query_attr_index, selectedIndex);
            }
        }
        // Return 2D Histogram
        else
        {
            if (((dims.size() == 0) || (mCubeDim < 3))&&(attrs.size()==2)) //if(dims.size()==0)
                return jointQuery(selectedIndex, attrs[0], attrs[1]);
            else if (((dims.size() == 0) || (mCubeDim < 3))&&(attrs.size()==3)) //if(dims.size()==0)
                return jointQuery(selectedIndex, attrs[0], attrs[1],attrs[2]);
            // query either 2D or 3D histogram
            else
            {
                int32_t src_attr_index = -1;
                int32_t src_attr_index2 = -1;
                int32_t selected_attr_index = -1;
                for (uint32_t ii = 0; ii < mAttrs.size(); ii++)
                {
                    if (mAttrs[ii] == attrs[0])
                        src_attr_index = ii;
                    if (mAttrs[ii] == attrs[1])
                        src_attr_index2 = ii;
                    if (mAttrs[ii] == dims[0])
                        selected_attr_index = ii;
                }

                uint32_t res = mDistributions[mSize - 1].get(dims[0]).resolution();

                std::vector<uint32_t> outHist(res * res, 0);

                std::vector<std::pair<float, float>> query_ranges = mDistributions[mSize - 1].get(dims[0]).ranges();

                float query_min = query_ranges[0].first;
                float query_max = query_ranges[0].second;

                float sel_query_min = ranges[0][0];
                float sel_query_max = ranges[0][1];

                if(sel_query_min<query_min)
                    sel_query_min = query_min;
                if(sel_query_max>query_max)
                    sel_query_max = query_max; 

                uint32_t start_ind = (uint32_t)(res * (sel_query_min - query_min) / (query_max - query_min));
                uint32_t end_ind = (uint32_t)(res * (sel_query_max - query_min) / (query_max - query_min));

                // Query for 2D or 3D depending on dims vs attrs

                if (src_attr_index == selected_attr_index)
                {
                    std::vector<uint32_t> src_hist = jointQuery(selectedIndex, attrs[0], attrs[1]);
                    for (uint32_t row = 0; row < res; row++)
                    {
                        for (uint32_t col = 0; col < res; col++)
                        {
                            if ((row >= start_ind) && (row <= end_ind))
                                outHist[row * res + col] = src_hist[row * res + col];
                        }
                    }
                    return interpolateHist(outHist, res, targetRes);
                }
                else if (src_attr_index2 == selected_attr_index)
                {
                    std::vector<uint32_t> src_hist = jointQuery(selectedIndex, attrs[0], attrs[1]);
                    for (uint32_t row = 0; row < res; row++)
                    {
                        for (uint32_t col = 0; col < res; col++)
                        {
                            if ((col >= start_ind) && (col <= end_ind))
                                outHist[row * res + col] = src_hist[row * res + col];
                        }
                    }
                    return interpolateHist(outHist, res, targetRes);
                }
                else
                {
                    // Get 3D Hist
                    if (selected_attr_index < src_attr_index)
                    {
                        std::vector<uint32_t> tempHist = jointQuery(selectedIndex, dims[0], attrs[0], attrs[1]);
                        res = mDistributions[mSize - 1].get(dims[0], attrs[0], attrs[1]).resolution();
                        outHist.resize(res * res, 0);
                        start_ind = (uint32_t)(res * (sel_query_min - query_min) / (query_max - query_min));
                        end_ind = (uint32_t)(res * (sel_query_max - query_min) / (query_max - query_min));
                        for (uint32_t row = 0; row < res; row++)
                        {
                            for (uint32_t col = 0; col < res; col++)
                            {
                                for (uint32_t lay = 0; lay < res; lay++)
                                {
                                    if ((row >= start_ind) && (row <= end_ind))
                                        outHist[col * res + lay] += tempHist[row * res * res + col * res + lay]; //src_hist[row*res+col];
                                }
                            }
                        }
                        std::vector<uint32_t> target_hist = jointQuery(selectedIndex, attrs[0], attrs[1]);                        
                        return interpolateHist(outHist, target_hist, res, targetRes);
                    }
                    else if (selected_attr_index > src_attr_index2)
                    {
                        std::vector<uint32_t> tempHist = jointQuery(selectedIndex, attrs[0], attrs[1], dims[0]);
                        res = mDistributions[mSize - 1].get(attrs[0], attrs[1], dims[0]).resolution();
                        outHist.resize(res * res, 0);
                        start_ind = (uint32_t)(res * (sel_query_min - query_min) / (query_max - query_min));
                        end_ind = (uint32_t)(res * (sel_query_max - query_min) / (query_max - query_min));
                        for (uint32_t row = 0; row < res; row++)
                        {
                            for (uint32_t col = 0; col < res; col++)
                            {
                                for (uint32_t lay = 0; lay < res; lay++)
                                {
                                    if ((lay >= start_ind) && (lay <= end_ind))
                                        outHist[row * res + col] += tempHist[row * res * res + col * res + lay]; //src_hist[row*res+col];
                                }
                            }
                        }
                        std::vector<uint32_t> target_hist = jointQuery(selectedIndex, attrs[0], attrs[1]);                        
                        return interpolateHist(outHist, target_hist, res, targetRes);
                    }
                    else
                    {
                        std::vector<uint32_t> tempHist = jointQuery(selectedIndex, attrs[0], dims[0], attrs[1]);
                        res = mDistributions[mSize - 1].get(attrs[0], dims[0], attrs[1]).resolution();
                        outHist.resize(res * res, 0);
                        start_ind = (uint32_t)(res * (sel_query_min - query_min) / (query_max - query_min));
                        end_ind = (uint32_t)(res * (sel_query_max - query_min) / (query_max - query_min));
                        for (uint32_t row = 0; row < res; row++)
                        {
                            for (uint32_t col = 0; col < res; col++)
                            {
                                for (uint32_t lay = 0; lay < res; lay++)
                                {
                                    if ((col >= start_ind) && (col <= end_ind))
                                        outHist[row * res + lay] += tempHist[row * res * res + col * res + lay]; //src_hist[row*res+col];
                                }
                            }
                        }
                        std::vector<uint32_t> target_hist = jointQuery(selectedIndex, attrs[0], attrs[1]);                        
                        return interpolateHist(outHist, target_hist, res, targetRes);
                    }
                }
            }
        }
    }
    else
    {
        if (dims.size() == 0)
        {
            std::string func_attr = mDistributions[mSize - 1].getAttr()[mFuncAttr];
            std::vector<std::string> func_vec;
            func_vec.push_back(func_attr);
            return jointQuery(selectedIndex, func_vec);
        }
        else
        { // ! When there is selection on other dim
            return functionQuery(dims, ranges, -1, selectedIndex);
        }
    }
}

std::vector<uint32_t> Selectivity::jointQuery(std::vector<uint32_t> &selectedIndex, std::string attr1, std::string attr2)
{
    uint32_t targetRes = mTargetRes;

    uint32_t res = mDistributions[0].get(attr1, attr2).resolution();

    std::vector<uint32_t> temp_hist(res * res, 0);

    for (uint32_t i = 0; i < (uint32_t)selectedIndex.size(); i++)
    {
        Histogram tempHist = mDistributions[selectedIndex[i]].get(attr1, attr2);
        //  Changing this in someway would speed up the process
        for (uint32_t ii = 0; ii < temp_hist.size(); ii++)
        {
            temp_hist[ii] += tempHist.data()[ii];
        }
    }
    return interpolateHist(temp_hist, res, targetRes);
}

std::vector<uint32_t> Selectivity::jointQuery(std::vector<uint32_t> &selectedIndex, std::string attr1, std::string attr2,
                                              std::string attr3)
{
    uint32_t targetRes = mTargetRes;

    uint32_t res = mDistributions[0].get(attr1, attr2, attr3).resolution();
    std::vector<uint32_t> temp_hist(res * res * res, 0);

    for (uint32_t i = 0; i < (uint32_t)selectedIndex.size(); i++)
    {
        Histogram tempHist = mDistributions[selectedIndex[i]].get(attr1, attr2, attr3);
        //  Changing this in someway would speed up the process
        for (uint32_t ii = 0; ii < temp_hist.size(); ii++)
        {
            temp_hist[ii] += tempHist.data()[ii];
        }
    }
    return temp_hist;
    //return interpolateHist(temp_hist, res, targetRes);
}

std::vector<uint32_t> Selectivity::jointQuery(std::string attr1, std::string attr2, std::string attr3)
{
    uint32_t targetRes = mTargetRes;

    uint32_t res = mDistributions[0].get(attr1, attr2, attr3).resolution();
    std::vector<uint32_t> temp_hist(res * res * res, 0);

    Histogram tempHist = mDistributions[mSize - 1].get(attr1, attr2, attr3);
    for (uint32_t ii = 0; ii < temp_hist.size(); ii++)
    {
        temp_hist[ii] += tempHist.data()[ii];
    }
    return temp_hist;
    //return interpolateHist(temp_hist, res, targetRes);
}
// ! Interpolation with uniform bucket assumption 
std::vector<uint32_t> Selectivity::interpolateHist(std::vector<uint32_t> &inputHist, uint32_t inputRes, uint32_t outputRes)
{
    if (inputRes >= outputRes)
        return inputHist;
    else
    {
        if (inputHist.size() == inputRes * inputRes)
        {
            std::vector<uint32_t> temp_hist(outputRes * outputRes, 0);
            uint32_t resDiff = outputRes / inputRes;
            for (uint32_t r = 0; r < inputRes; r++)
            {
                for (uint32_t c = 0; c < inputRes; c++)
                {   uint32_t totalfreq = 0;
                    uint32_t freq = inputHist[r * inputRes + c];
                    // uint32_t freq = (uint32_t)ceil((double)freq/resDiff/resDiff);
                    for (uint32_t rr = 0; rr < resDiff; rr++)
                    {
                        for (uint32_t cc = 0; cc < resDiff; cc++)
                        {
                            temp_hist[(resDiff * r + rr) * outputRes + (resDiff * c + cc)] = freq / resDiff / resDiff;
                            totalfreq+=temp_hist[(resDiff * r + rr) * outputRes + (resDiff * c + cc)];
                        }
                    }
                    temp_hist[(resDiff*r)*outputRes + resDiff*c] += freq-totalfreq;//freq-(freq/resDiff/resDiff)*(resDiff*resDiff-1); //freq1; 
                    //temp_hist[resDiff*r*outputRes+resDiff*c] = freq-freq*(resDiff*resDiff-1)/resDiff/resDiff;
                }
            }
            return temp_hist;
        }
        else
        {
            std::vector<uint32_t> temp_hist(outputRes, 0);
            uint32_t resDiff = outputRes / inputRes;
            for (uint32_t r = 0; r < inputRes; r++)
            {
                uint32_t freq = inputHist[r];
                uint32_t totalfreq = 0;
                // uint32_t freq1 = (uint32_t)ceil((double)freq/resDiff);

                for (uint32_t rr = 0; rr < resDiff; rr++)
                {
                    temp_hist[resDiff * r + rr] = freq / resDiff;
                    //temp_hist[resDiff*r] = freq-freq*(resDiff-1)/resDiff;
                    totalfreq+=temp_hist[resDiff * r + rr];
                }
                // temp_hist[resDiff * r] = freq1;
                temp_hist[resDiff*r] +=freq-totalfreq; //freq-freq/resDiff*(resDiff-1);//freq1;

            }
            return temp_hist;
        }
    }
}

// ! Interpolation with higher resolution lower dimensional projections 
std::vector<uint32_t> Selectivity::interpolateHist(std::vector<uint32_t> &inputHist, std::vector<uint32_t> &targetHist, 
                                                   uint32_t inputRes, uint32_t outputRes)
{
    if (inputRes >= outputRes)
        return inputHist;
    else
    {
        if (inputHist.size() == inputRes * inputRes)
        {
            std::vector<uint32_t> temp_hist(outputRes * outputRes, 0);
            uint32_t resDiff = outputRes / inputRes;
            for (uint32_t r = 0; r < inputRes; r++)
            {
                for (uint32_t c = 0; c < inputRes; c++)
                {
                    uint32_t freq = inputHist[r * inputRes + c];
                    uint32_t old_freq = 0;
                    for (uint32_t rr = 0; rr < resDiff; rr++)
                    {
                        for (uint32_t cc = 0; cc < resDiff; cc++)
                        {
                            temp_hist[(resDiff * r + rr) * outputRes + (resDiff * c + cc)] = freq*targetHist[(resDiff * r + rr) * outputRes + (resDiff * c + cc)];//freq / resDiff / resDiff;
                            old_freq+=targetHist[(resDiff * r + rr) * outputRes + (resDiff * c + cc)];
                        }
                    }
                    if(old_freq!=0)
                    {   //uint32_t freq1 = temp_hist[(resDiff * r) * outputRes + (resDiff * c)];
                        uint32_t totalfreq = 0;
                        for (uint32_t rr = 0; rr < resDiff; rr++)
                        {
                            for (uint32_t cc = 0; cc < resDiff; cc++)
                            {
                                temp_hist[(resDiff * r + rr) * outputRes + (resDiff * c + cc)]/= old_freq;
                                totalfreq+=temp_hist[(resDiff * r + rr) * outputRes + (resDiff * c + cc)];
                            }
                        }
                        
                        temp_hist[(resDiff * r) * outputRes + (resDiff * c)] += freq-totalfreq;
                    
                    }
                    //temp_hist[resDiff*r*outputRes+resDiff*c] = freq-freq*(resDiff*resDiff-1)/resDiff/resDiff;
                }
            }
            return temp_hist;
        }
        else
        {
            std::vector<uint32_t> temp_hist(outputRes, 0);
            uint32_t resDiff = outputRes / inputRes;
            for (uint32_t r = 0; r < inputRes; r++)
            {
                uint32_t freq = inputHist[r];
                uint32_t old_freq = 0;
                for (uint32_t rr = 0; rr < resDiff; rr++)
                {
                    temp_hist[resDiff * r + rr] = freq*targetHist[resDiff * r + rr]; // resDiff;
                    old_freq+=targetHist[resDiff * r + rr];
                    //temp_hist[resDiff*r] = freq-freq*(resDiff-1)/resDiff;
                }
                if(old_freq!=0)
                {   uint32_t totalfreq = 0;
                    for (uint32_t rr = 0; rr < resDiff; rr++)
                    {
                        temp_hist[resDiff * r + rr] /= old_freq;//freq*targetHist[resDiff * r + rr]; // resDiff;
                        totalfreq+= temp_hist[resDiff * r + rr];
                        //old_freq+=targetHist[resDiff * r + rr];
                        //temp_hist[resDiff*r] = freq-freq*(resDiff-1)/resDiff;
                    }
                    temp_hist[resDiff * r] += freq-totalfreq;
                }
            }
            return temp_hist;
        }
    }
}


// Helper function for cube query? 
// input cube, cube dimension, cube res, range for each dimension (or selected range? ), output index? 

// std::vector<uint32_t> Selectivity::cubeQuery(std::vector<uint32_t>& inputCube, uint32_t cubeDim, uint32_t res, 
//                                              std::vector<std::vector<float> >ranges, std::vector<uint32_t> outDims)
// {


// }
