/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}


void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}


vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnLoopQuery!=pKF->mnId)
                {
                    pKFi->mnLoopWords=0;
                    if(!spConnectedKeyFrames.count(pKFi))
                    {
                        pKFi->mnLoopQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++;
            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

// DetectRelocalizationCandidates(&mCurrentFrame)
// 현재 frame과 word를 공유하는 모든 keyframe을 찾는 함수 -> BoW의 첫 번째 기능(BoW vector 이용)
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex); // unique_lock class의 객체인 lock은 mutex 객체인 mMutex를 소유한다.

        // DBoW2::BowVector -> std::map<WordId, WordValue>
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first]; // 특정한 word id에 해당하는 keyframes -> lKFs

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit; // de-reference -> 특정한 word id에 해당하는 keyframe
                // Q. 특정 frame이 처음으로 BoW에 접근하였다면,
                if(pKFi->mnRelocQuery!=F->mnId) // 특정한 word id에 해당하는 keyframe의 mnRelocQuery가 frame의 id와 일치하지 않는다면,
                {
                    pKFi->mnRelocWords=0; // 특정한 word id에 해당하는 keyframe의 mnRelocWords를 초기화
                    pKFi->mnRelocQuery=F->mnId; // frame의 id -> 특정한 word id에 해당하는 keyframe의 mnRelocQuery
                    lKFsSharingWords.push_back(pKFi); // 특정한 word id에 해당하는 keyframe -> lKFsSharingWords list
                }
                pKFi->mnRelocWords++; // mnRelocWords -> 해당 keyframe과 현재 frame이 공유하는 relocalization words의 개수
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    // lKFsSharingWords list -> 특정한 word를 공유하는 keyframe
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords) // keyframe->mnRelocWords : 해당 keyframe과 현재 frame이 공유하는 relocalization words의 개수
            maxCommonWords=(*lit)->mnRelocWords; // keyframe->mnRelocWords -> maxCommonWords : 각 keyframe과 현재 frame이 공유하는 가장 많은 relocalization words 개수
    }

    int minCommonWords = maxCommonWords*0.8f; // threshold

    // pair class는 사용자가 지정한 2개 타입의 데이터를 저장하는데 사용한다.
    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    // lKFsSharingWords list -> 특정한 word를 공유하는 keyframe
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords) // 해당 keyframe과 현재 frame이 공유하는 relocalization words의 개수가 일정 threshold(maxCommonWords*0.8)를 넘으면,
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec); // 해당 keyframe과 현재 frame의 similarity score 계산
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi)); // keyframe + 해당 keyframe의 similarity score
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second; // it->second : keyframe
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10); // 해당 keyframe의 covisibility graph 상에서의 N개의 neighbors 추출

        float bestScore = it->first; // it->first : 해당 keyframe의 similarity score
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++) // 해당 keyframe의 covisibility graph 상에서 N개의 neighbors만큼 반복
        {
            KeyFrame* pKF2 = *vit; // neighbors keyframe
            if(pKF2->mnRelocQuery!=F->mnId) // Q.
                continue; // 해당 for문을 빠져나가라.

            accScore+=pKF2->mRelocScore; // neighbors keyframe의 mRelocScore까지 고려
            // 현재 frame과 word를 공유하는 keyframe의 neighbors 중, 가장 많은 word를 공유하는 keyframe을 찾는다.
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF)); // 현재 frame과 word를 공유하는 keyframe의 neighbors 중, 가장 많은 word를 공유하는 keyframe
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    // set container : set container는 노드 기반 컨테이너이며 균형 이진트리로 구현되어 있다. key라 불리는 원소들의 집합으로 이루어진 container이다. 
    // key 값은 중복이 허용되지 않는다. 원소가 insert 멤버 함수에 의해 삽입이 되면, 원소는 자동으로 정렬된다.
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    // 현재 frame과 word를 공유하는 keyframe의 neighbors 중, 가장 많은 word를 공유하는 keyframe + 해당 keyframe의 similarity score
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first; // it->first : similarity score
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second; // it->second : neighbors keyframe
            if(!spAlreadyAddedKF.count(pKFi)) // 해당 keyframe이 spAlreadyAddedKF에 존재하지 않는다면 -> 중복 허용 x
            {
                vpRelocCandidates.push_back(pKFi); // pKFi(neighbors keyframe) -> vpRelocCandidates
                spAlreadyAddedKF.insert(pKFi); // pKFi(neighbors) -> spAlreadyAddedKF
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
