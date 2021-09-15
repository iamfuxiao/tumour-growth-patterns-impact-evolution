#ifndef TUMOUR_GROWTH_PATTERNS_CUH_INCLUDED
#define TUMOUR_GROWTH_PATTERNS_CUH_INCLUDED

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#include <random>
#include <algorithm>
#include <unistd.h> // getpid()
#include <unordered_map>

using namespace std;

// system setting (NTHR = 1024 for cluster; 128 for laptop)
#define N 200
#define T 601
#define SIZE_MAX 1e6
#define FOUTPUT T-1
#define FOUTPUT_SURF 10
#define NTHR 1024
const pid_t PROC_ID               =   getpid();

// tumour essential
const char typeGrowthMode = 's';    // 's' for surface; 'v' for volume
const char typeDriverAdvantage = 's'; // 's' for saturated; 'a' for additive
const bool flagSaveCellDynamicsOverTime = false;    // false for replicate runs
const bool flagTumourApop = false;
const bool flagTumourNecr = true;

// tumour ... founder cell location
const int FOUNDER_X = N/2-1;
const int FOUNDER_Y = N/2-1;
const int FOUNDER_Z = N/2-1;

// tumour ... growth (probability of copying)
const float P_COPY = 0.25; // probability of voxel duplication

// tumour ... death (probability of emptying)
const float P_EMPTY = 0.05; //
const int N_START_EMPTY = 100; // minimum number of voxels to start apoptosis

// clone
const string typeCloneCreate = "emergeRCC";
const short N_EVENT_DRIVER_RCC = 26;
const float P_EVENT_DRIVER_RCC_UPON_PROLIF = 2e-4;  // 1e-3 for Volume Growth; 2e-4 for Surface Growth

// dictionary recording event name and their level
const string arrMutaDriverNameRCC[N_EVENT_DRIVER_RCC] {
    "ARID1A", "BAP1", "KDM5C", "MTOR", "PBRM1", "PIK3CA",
    "PTEN", "SETD2", "TP53", "TSC1", "TSC2", "VHL",
    "gain_1q", "gain_2q", "gain_5q", "gain_7q", "gain_8q", "gain_12p", "gain_20q",
    "loss_1p", "loss_3p", "loss_4q", "loss_6q", "loss_8p", "loss_9p", "loss_14q"
};
// ... mutations that enhance scnas
const short N_MUTA_INCREASE_SCNA_RCC = 2;
const string arrMutaIncreaseScnaRCC[N_MUTA_INCREASE_SCNA_RCC] {
    "BAP1", "PBRM1"
};

// ... driver advantage in proliferation
// ... ... saturated model

// estimated according to Ki67 ranking ... the scaling factor for P_COPY
// ... this is used only in saturated model of driver advantage
const float driverProlLv1 = 2;
const float driverProlLv2 = 4;
// estimated according to ordering of events .. the scaling factor for P_EVENT_DRIVER_RCC
const float driverEmerLv1 = 1e-3;

// rationale:
// (1) assume all gene mutations are equally advantageous (balancing Ki67 ranking & 2018 paper prevalence)
// (2) assume SCNAs are more advantageous
const unordered_map<string, float> umapDriverProlRCC = {
    {"ARID1A", 1}, {"BAP1", 1}, {"KDM5C", 1}, {"MTOR", 1}, {"PBRM1", 1}, {"PIK3CA", 1},
    {"PTEN", 1}, {"SETD2", 1}, {"TP53", 1}, {"TSC1", 1}, {"TSC2", 1}, {"VHL", 1},
    {"gain_1q", 1}, {"gain_2q", driverProlLv1}, {"gain_5q", driverProlLv1},
    {"gain_7q", driverProlLv2}, {"gain_8q", driverProlLv1}, {"gain_12p", driverProlLv1},
    {"gain_20q", driverProlLv2}, {"loss_1p", driverProlLv1}, {"loss_3p", 1}, {"loss_4q", driverProlLv2},
    {"loss_6q", 1}, {"loss_8p", driverProlLv2}, {"loss_9p", driverProlLv1}, {"loss_14q", driverProlLv1}
};

// ... ... additive model
// version 1 -- (mean selection coef = 0.03)
/*
const unordered_map<string, float> umapDriverProlAdditiveRCC = {
    {"ARID1A", 0.006}, {"BAP1", 0.022}, {"KDM5C", 0.014}, {"MTOR", 0.046}, {"PBRM1", 0.024}, {"PIK3CA", 0.05},
    {"PTEN", 0.012}, {"SETD2", 0.028}, {"TP53", 0.056}, {"TSC1", 0.034}, {"TSC2", 0.008}, {"VHL", 0},
    {"gain_1q", 0.018}, {"gain_2q", 0.026}, {"gain_5q", 0.032}, {"gain_7q", 0.054},
    {"gain_8q", 0.04}, {"gain_12p", 0.036}, {"gain_20q", 0.044},
    {"loss_1p", 0.03}, {"loss_3p", 0}, {"loss_4q", 0.052}, {"loss_6q", 0.01},
    {"loss_8p", 0.048}, {"loss_9p", 0.038}, {"loss_14q", 0.042}
};
*/
// version 2 -- (mean selection coef = 0.15)
/*
const unordered_map<string, float> umapDriverProlAdditiveRCC = {
    {"ARID1A", 0.03}, {"BAP1", 0.11}, {"KDM5C", 0.07},
    {"MTOR", 0.23}, {"PBRM1", 0.12}, {"PIK3CA", 0.25},
    {"PTEN", 0.06}, {"SETD2", 0.14}, {"TP53", 0.28},
    {"TSC1", 0.17}, {"TSC2", 0.04}, {"VHL", 0},
    {"gain_1q", 0.09}, {"gain_2q", 0.13}, {"gain_5q", 0.16}, {"gain_7q", 0.27},
    {"gain_8q", 0.2}, {"gain_12p", 0.18}, {"gain_20q", 0.22},
    {"loss_1p", 0.15}, {"loss_3p", 0}, {"loss_4q", 0.26}, {"loss_6q", 0.05},
    {"loss_8p", 0.24}, {"loss_9p", 0.19}, {"loss_14q", 0.21}
};
*/
// version 3 -- (mean selection coef = 0.1)
const unordered_map<string, float> umapDriverProlAdditiveRCC = {
    {"ARID1A", 0.015}, {"BAP1", 0.055}, {"KDM5C", 0.035},
    {"MTOR", 0.115}, {"PBRM1", 0.06}, {"PIK3CA", 0.125},
    {"PTEN", 0.03}, {"SETD2", 0.07}, {"TP53", 0.14},
    {"TSC1", 0.085}, {"TSC2", 0.02}, {"VHL", 0},
    {"gain_1q", 0.045}, {"gain_2q", 0.065}, {"gain_5q", 0.08}, {"gain_7q", 0.135},
    {"gain_8q", 0.1}, {"gain_12p", 0.09}, {"gain_20q", 0.11},
    {"loss_1p", 0.075}, {"loss_3p", 0}, {"loss_4q", 0.13}, {"loss_6q", 0.025},
    {"loss_8p", 0.12}, {"loss_9p", 0.095}, {"loss_14q", 0.105}
};

// ... driver advantage in emergence probability
const unordered_map<string, float> umapDriverEmerRCC = {
    {"ARID1A", 1}, {"BAP1", 1}, {"KDM5C", 1}, {"MTOR", 1},
    {"PBRM1", 1}, {"PIK3CA", 1}, {"PTEN", 1}, {"SETD2", 1},
    {"TP53", 1}, {"TSC1", 1}, {"TSC2", 1}, {"VHL", 1},
    {"gain_1q", driverEmerLv1}, {"gain_2q", driverEmerLv1}, {"gain_5q", driverEmerLv1},
    {"gain_7q", driverEmerLv1}, {"gain_8q", driverEmerLv1}, {"gain_12p", driverEmerLv1},
    {"gain_20q", driverEmerLv1}, {"loss_1p", driverEmerLv1}, {"loss_3p", driverEmerLv1}, {"loss_4q", driverEmerLv1},
    {"loss_6q", driverEmerLv1}, {"loss_8p", driverEmerLv1}, {"loss_9p", driverEmerLv1}, {"loss_14q", driverEmerLv1}
};

// -------------------------------------------- //

// necrosis ;
const float NECROSIS_DIST_FROM_SURFACE = 15;
const float P_NECROSIS = 0.5;

// search range (need to shuffle this before sending to device)
#define nb26_size 26
//__constant__ short nb26[nb26_size][3];
const vector<vector<int>> nb26 {
    {-1,-1,-1}, {-1,-1,0}, {-1,-1,1},
    {-1,0,-1}, {-1,0,0}, {-1,0,1},
    {-1,1,-1}, {-1,1,0}, {-1,1,1},
    {0,-1,-1}, {0,-1,0}, {0,-1,1},
    {0,0,-1}, {0,0,1},
    {0,1,-1}, {0,1,0}, {0,1,1},
    {1,-1,-1}, {1,-1,0}, {1,-1,1},
    {1,0,-1}, {1,0,0}, {1,0,1},
    {1,1,-1}, {1,1,0}, {1,1,1}
};

// objects
class Cell
{
    int id;
    vector<int> loc;
    bool alive;

public:
    Cell ();
    Cell (int id, bool alive, vector<int> loc);
    ~Cell ();

    // setters
    __host__ void set_id(int id) {this->id=id;}
    __host__ void set_alive(bool alive) {this->alive=alive;}
    __host__ void set_loc(vector<int> loc) {this->loc=loc;}
    //__host__ void set_immuResp(char immuResp) {this->immuResp=immuResp;}

    // getters
    __host__ int get_id() const {return this->id;}
    __host__ bool get_alive() const {return this->alive;}
    __host__ vector<int> get_loc() const {return this->loc;}
    //__host__ char get_immuResp() const {return this->immuResp;}
};

// gpu kernel functions
__global__ void growth_random_kernel(char * tumour, long * clone,
                                     int * sites_valid, float * sites_valid_pcopy, int n_sites_valid,
                                     int * sites_new, bool* permit_nb26_d, short * ptr_nb26_d,
                                     float * ra01);
__global__ void update_surface_kernel(char * tumour, bool * surface, short * ptr_nb26_d);
__global__ void necrosis_kernel(bool * near_surface, int n_sites_valid, int * sites_valid, int n_voxSurf, short * ptr_voxSurf_d);

// cpu functions
void emerge_subclones_rcc_uponProlif(long * clone, vector<Cell> &vecCell, vector<int> vecNewCellID,
                        unordered_map <long, vector<string>> &umapCloneEvents, unordered_map <long, long> &umapCloneEventsOrder);
void growth2(char * tumour, int * cellids, long * clone, vector<Cell> & vecCell,
    unordered_map <long, vector<string>> &umapCloneEvents,
    unordered_map <long, long> &umapCloneEventsOrder);
void growth2_volume(char * tumour, int * cellids, long * clone, vector<Cell> & vecCell,
             unordered_map <long, vector<string>> &umapCloneEvents,
             unordered_map <long, long> &umapCloneEventsOrder);

void necrosis2(char * tumour, int * cellids, long * clone, vector<Cell> & vecCell);
void update_surface2(char * tumour, vector<vector<int>> & vecVoxSurf);

void writeDynamics(char * tumour, long *clone, int *cellids,
                   unordered_map<long, vector<string>> &umapCloneEvents, unordered_map <long, long> &umapCloneEventsOrder, int t_now,
               bool flagWriteCoord, bool flagIsFinal);


#endif // TUMOUR_GROWTH_PATTERNS_CUH_INCLUDED
