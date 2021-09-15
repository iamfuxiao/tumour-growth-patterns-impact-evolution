// Note that "Cell" or "cell" is a coarse-grained representation in this work
#include "tumour_growth_patterns.cuh"

/* ---------------- Cells ------------------ */
Cell::Cell() {}
Cell::Cell(int id, bool alive, vector<int> loc)
{
    this->id = id;
    this->alive = alive;
    this->loc = loc;
}
Cell::~Cell() {}

int main(int argc, char** argv)
{
    // [1] allocate memory for tumour & clone
    // ... tumour: can refer to proliferative status 'p' (proliferative) or 'n' (non-proliferative), 'e' (empty sites), 'b' (barrier/non-permissive sites) or 'd' (dead voxels)
    char * tumour;
    // ... cellids: cell ids
    int * cellids;
    // ... clone: integers indicate the subclone-specific ids
    // ... ... if typeCloneCreate == "emergeRCC", this reflects the a list of N_MUTA_DRIVER mutations that are recorded separately
    long * clone;
    unordered_map <long, vector<string>> umapCloneEvents; // clone id : vector of events
    unordered_map <long, long> umapCloneEventsOrder; // child clone id : parent clone id

    vector<Cell> vecCell;

    // ... Allocate Unified Memory -- accessible from CPU or GPU
    cudaError_t r   = cudaMallocManaged(&tumour, N*N*N*sizeof(char));
    cudaError_t rr  = cudaMallocManaged(&clone, N*N*N*sizeof(long));
    cudaError_t rrr = cudaMallocManaged(&cellids, N*N*N*sizeof(int));
    if (r != cudaSuccess)
    {
        printf("CUDA Error on %s\n", cudaGetErrorString(r));
        exit(0);
    } else {
        size_t total_mem, free_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << " Currently " << free_mem << " bytes free" << std::endl;
    }

    // [2] initialise the first cell
    cout << "> initialising first cancer cell... " << endl;
    int ci = FOUNDER_X, cj = FOUNDER_Y, ck = FOUNDER_Z;
    for (int i = 0; i < N; i ++)
    {
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k ++)
            {
                *(tumour + i*N*N + j*N + k) = 'e';   // initialise all sites as empty
            }
        }
    }
    *(tumour  + ci*N*N + cj*N + ck) = 'p';    // set the founder cell to be proliferative
    *(clone   + ci*N*N + cj*N + ck) = 0;      // parent clone
    *(cellids + ci*N*N + cj*N + ck) = 0;      // first cell
    vector<int> loc {ci,cj,ck};

    if (typeCloneCreate == "emergeRCC")
    {
        vector<string> cloneEvents {"loss_3p", "VHL"};  // assumed as clonal events
        umapCloneEvents.insert ({  {0, cloneEvents}  });  // create clone id 0 harbouring clonal events
    }

    bool alive = true;
    Cell c = Cell(0, alive, loc);  // create the first cell
    vecCell.push_back(c);   // add the cell to the vector of cells
    cout << " done! " << endl;


    // [3] main loop of simulation
    cout << "> main loop of simulations < " << endl;

    short tall = (short)T;
    short t = 0;
    while (t <= tall)
    {
        cout << "\n... time : " << t << endl;

        // simulate growth
        if (true)
        {
            cout << "----------------------------------------------" << endl;
            cout << "---------------     GROWTH     ----------------" << endl;
            cout << "----------------------------------------------" << endl;
            if (typeGrowthMode == 's')  // Surface Growth
              growth2(tumour, cellids, clone, vecCell, umapCloneEvents, umapCloneEventsOrder);
            if (typeGrowthMode == 'v')  // Volume Growth
              growth2_volume(tumour, cellids, clone, vecCell, umapCloneEvents, umapCloneEventsOrder);
        }

        // simulate necrosis
        if (flagTumourNecr)
        {
            cout << "----------------------------------------------" << endl;
            cout << "---------------     NECROSIS    ----------------" << endl;
            cout << "----------------------------------------------" << endl;
            necrosis2(tumour, cellids, clone, vecCell);
        }

        if (t % FOUTPUT_SURF == 0)
        {
            cout << "> write Cell Dynamics output file <" << endl;

            bool flagWriteCoord = flagSaveCellDynamicsOverTime;
            bool flagIsFinal = false;
            writeDynamics(tumour, clone, cellids, umapCloneEvents, umapCloneEventsOrder, t, flagWriteCoord, flagIsFinal);
            cout << "__DONE__" << endl;
        }

        // ======== stop simulation by tumour size ========
        // ... check tumour size = number of voxels with 'p' or 'n' or 'd' status
        // ... if size exceeds SIZE_MAX, save information
        // ... break the loop!
        int size_now = 0;
        for (int ii = 0; ii < N; ii ++)
        {
            for (int jj = 0; jj < N; jj ++)
            {
                for (int kk = 0; kk < N; kk ++)
                {
                    char status_this = *(tumour + ii*N*N + jj*N + kk);
                    if (status_this == 'p' || status_this == 'n' || status_this == 'd')
                        size_now ++;
                }
            }
        }
        if (size_now >= SIZE_MAX)
        {
            cout << "> write **FINAL** Cell Dynamics output file  <" << endl;

            bool flagWriteCoord = true;
            bool flagIsFinal = true;
            writeDynamics(tumour, clone, cellids, umapCloneEvents, umapCloneEventsOrder, t, flagWriteCoord, flagIsFinal);
            cout << "__DONE__" << endl;

            break;
        }


        // ================================================

        t++;
    }

    // [6] de-allocate memory
    cudaFree(tumour);
    cudaFree(cellids);
    cudaFree(clone);

    return 0;
}


// GROWTH: randomly access the whole lattice; division of labor by number of sites_valid
__global__ void growth_random_kernel(char * tumour, long * clone,
                                     int * sites_valid, float * sites_valid_pcopy, int n_sites_valid,
                                     int * sites_new, bool* permit_nb26_d, short * ptr_nb26_d,
                                     float * ra01)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_sites_valid)
    {
        float * ptr_ra = ra01 + tid;
        float * ptr_pcopy = sites_valid_pcopy + tid;

        *(sites_new+3*tid)   = -1;  // indicate no growth
        *(sites_new+3*tid+1) = -1;
        *(sites_new+3*tid+2) = -1;

        if (*ptr_ra < *ptr_pcopy)
        {
            int * ptr = sites_valid + tid*3;
            short di, dj, dk;
            bool the_permit;
            int i = *ptr, j = *(ptr+1), k = *(ptr+2);

            char status = *(tumour + i*N*N + j*N + k);
            long subcl = *(clone  + i*N*N + j*N + k);

            bool flagGrowth = false;

            if (status == 'p')
            {
                for (short inb = 0; inb < nb26_size; inb ++)  // WARNING: this should be shuffled
                {
                    di = *(ptr_nb26_d+3*inb);
                    dj = *(ptr_nb26_d+3*inb+1);
                    dk = *(ptr_nb26_d+3*inb+2);

                    if (i+di < 0 || i+di > N-1 ||
                        j+dj < 0 || j+dj > N-1 ||
                        k+dk < 0 || k+dk > N-1)
                        continue;

                    the_permit = *(permit_nb26_d + inb);
                    if (the_permit == false)
                        continue;

                    char status2 = *(tumour + (i+di)*N*N + (j+dj)*N + (k+dk));

                    if (status2 == 'e' || status2 == 'd')   // directly update DIFFERENT memory of sites_new!
                    {
                        flagGrowth = true;
                        *(tumour + (i+di)*N*N + (j+dj)*N + (k+dk)) = 'p';
                        *(clone  + (i+di)*N*N + (j+dj)*N + (k+dk)) = subcl; // copy clone identity
                        *(sites_new+3*tid)   = i+di;
                        *(sites_new+3*tid+1) = j+dj;
                        *(sites_new+3*tid+2) = k+dk;

                        break;
                    }

                }

                if (flagGrowth == false)
                    *(tumour + i*N*N + j*N + k) = 'n';
            }
        }
    }

}

// UPDATE TUMOUR SURFACE
__global__ void update_surface_kernel(char * tumour, bool * surface, short * ptr_nb26_d)
{
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N*N*N)
    {
        int i = tid / N/N;
        int j = (tid - i*N*N)/N;
        int k = tid - i*N*N - j*N;

        bool is_surface_voxel = false;

        char status = *(tumour + i*N*N + j*N + k);
        if (status == 'p' || status == 'n')
        {
            for (short inb = 0; inb < nb26_size; inb ++)
            {
                short di = *(ptr_nb26_d+3*inb);
                short dj = *(ptr_nb26_d+3*inb+1);
                short dk = *(ptr_nb26_d+3*inb+2);

                int ivox_nb = (i+di)*N*N + (j+dj)*N + k+dk;
                char status_nb = *(tumour + ivox_nb);

                if (status_nb == 'e' || status_nb == 'b')
                {
                    is_surface_voxel = true;
                    break;
                }
            }
        }

        *(surface+i*N*N+j*N+k) = is_surface_voxel;
    }
}

// NECROSIS:
__global__ void necrosis_kernel(bool * near_surface, int n_sites_valid, int * sites_valid, int n_voxSurf, short * ptr_voxSurf_d)
{
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_sites_valid)
    {
        int i = *(sites_valid+3*tid);
        int j = *(sites_valid+3*tid+1);
        int k = *(sites_valid+3*tid+2);

        bool is_near_surface = false;

        for (short inb = 0; inb < n_voxSurf; inb ++)
        {
            int i2 = *(ptr_voxSurf_d+3*inb);
            int j2 = *(ptr_voxSurf_d+3*inb+1);
            int k2 = *(ptr_voxSurf_d+3*inb+2);

            if (abs(i-i2) >= NECROSIS_DIST_FROM_SURFACE ||
            abs(j-j2) >= NECROSIS_DIST_FROM_SURFACE ||
            abs(k-k2) >= NECROSIS_DIST_FROM_SURFACE)
                continue;

            float dsq = (i-i2)*(i-i2) + (j-j2)*(j-j2) + (k-k2)*(k-k2);

            if (dsq < NECROSIS_DIST_FROM_SURFACE*NECROSIS_DIST_FROM_SURFACE)
            {
                is_near_surface = true;
                break;
            }
        }

        *(near_surface+i*N*N+j*N+k) = is_near_surface;

    }
}


// TUMOUR GROWTH growth2()
void growth2(char * tumour, int * cellids, long * clone, vector<Cell> & vecCell,
             unordered_map <long, vector<string>> &umapCloneEvents,
             unordered_map <long, long> &umapCloneEventsOrder)
{
    // -------------------------------------------------------
    // [1] shuffle vecCell
    // -------------------------------------------------------
    cout << "[1] shuffle vecCell : random_shuffle" << endl;

    vector<Cell> vecCellShu = vecCell;
    random_shuffle ( vecCellShu.begin(), vecCellShu.end() );

    // -------------------------------------------------------
    // [2] check cell death & the number of alive cells
    // -------------------------------------------------------
    cout << "[2] check cell death & the number of alive cells" << endl;
    vector<Cell> vecCellShuValid;
    // ... prepare random numbers for cell death probability
    int n_sites_is_cell = vecCellShu.size();
    float * ra01, * ptr_ra;
    cudaError_t rrrr = cudaMallocManaged(&ra01, n_sites_is_cell*sizeof(float));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);   //uniform distribution between 0 and 1
    ptr_ra = ra01;
    for (int nra = 0; nra < n_sites_is_cell; ++ nra) {
         * ptr_ra = dis(gen);
         ptr_ra ++;
    }
    // ... push_back Cells into the vecCellShuValid vector
    ptr_ra = ra01;
    for (vector<Cell>::iterator it_C = vecCellShu.begin(); it_C != vecCellShu.end(); it_C ++)
    {
        bool alive = it_C->get_alive();
        if (alive)
        {
            // consider cell death here
            float p_empty = P_EMPTY;

            if (flagTumourApop && *ptr_ra < p_empty
                && vecCellShu.size() > N_START_EMPTY)   // selected to undergo death
            {
                int cid_e = it_C->get_id();
                vecCell[cid_e].set_alive(false);
                vector<int> loc_e = it_C->get_loc();
                *(tumour+loc_e[0]*N*N+loc_e[1]*N+loc_e[2]) = 'd';   // set the site to 'd' status
                ptr_ra ++;

                // change proliferative status to 'p' for the surrounding cells of the dying cell
                int dx, dy, dz;
                for (int ss = 0; ss < nb26_size; ss ++)
                {
                    dx = nb26[ss][0];
                    dy = nb26[ss][1];
                    dz = nb26[ss][2];
                    vector<int> loc_n {loc_e[0]+dx, loc_e[1]+dy, loc_e[2]+dz};
                    if ( *(tumour+loc_n[0]*N*N+loc_n[1]*N+loc_n[2]) == 'n' )    // non-proliferative
                        *(tumour+loc_n[0]*N*N+loc_n[1]*N+loc_n[2]) = 'p';
                }

                continue;
            }

            // check if it's proliferative
            vector<int> loc_c = it_C->get_loc();
            if (*(tumour+loc_c[0]*N*N+loc_c[1]*N+loc_c[2]) == 'p')
                vecCellShuValid.push_back(*it_C);   // add this cell to a vector pending probabilistic proliferation
        }
    }
    cudaFree(ra01);


    // -------------------------------------------------------
    // [3] allocate memory for valid sites
    // -------------------------------------------------------
    int * sites_valid;
    int * sites_new;
    float * sites_valid_pcopy;
    int n_sites_valid = vecCellShuValid.size();

    cout << "[3] allocate memory for valid sites : cudaMallocManaged" << endl;
    cudaError_t r2 = cudaMallocManaged(&sites_valid, n_sites_valid*3*sizeof(int));
    cudaError_t r3 = cudaMallocManaged(&sites_new, n_sites_valid*3*sizeof(int));
    cudaError_t r5 = cudaMallocManaged(&sites_valid_pcopy, n_sites_valid*sizeof(float));
    cout << "     n_sites_valid = " << n_sites_valid << endl;

    // -------------------------------------------------------
    // [4] define valid_sites according to vecCellShuValid
    // -------------------------------------------------------
    cout << "[4] copy alive cell loc to valid sites : memcpy" << endl;
    int * ptr = sites_valid;
    float * ptr_pcopy = sites_valid_pcopy;

    for (vector<Cell>::iterator it_C = vecCellShuValid.begin(); it_C != vecCellShuValid.end(); it_C ++)
    {
        vector<int> loc = it_C->get_loc();
        memcpy(ptr, &loc[0], 3*sizeof(int));
        ptr += 3;

        // ... assign P_COPY * scale_factor to this site
        float scale_factor = 1;

        if (typeCloneCreate == "emergeRCC")
        {
            // find the cloneEvents in this clone
            long subcl = *(clone  + loc[0]*N*N + loc[1]*N + loc[2]);
            vector<string> cloneEvents = umapCloneEvents[subcl];

            // ... saturated model of driver advantage
            if (typeDriverAdvantage == 's')
            {
                for (vector<string>::iterator it_v = cloneEvents.begin(); it_v != cloneEvents.end(); it_v++)
                {
                    string event = *(it_v);
                    float event_scale_factor = umapDriverProlRCC.at(event);
                    if (event_scale_factor > scale_factor)
                        scale_factor = event_scale_factor;
                }
            }

            // ... additive model of driver advantage
            if (typeDriverAdvantage == 'a')
            {
                float total_selection_coef = P_COPY;
                for (vector<string>::iterator it_v = cloneEvents.begin(); it_v != cloneEvents.end(); it_v++)
                {
                    string event = *(it_v);
                    float event_selection_coef = umapDriverProlAdditiveRCC.at(event);
                    total_selection_coef += event_selection_coef;

                }
                scale_factor = total_selection_coef / P_COPY;
            }

        }

        *ptr_pcopy = P_COPY * scale_factor;
        ptr_pcopy ++;
    }

    // -------------------------------------------------------
    // [5] shuffle neighbourhood
    // -------------------------------------------------------
    cout << "[5] shuffle neighbourhood nb26 : random_shuffle" << endl;
    const float P_SCALE_2ND = 0.3;  // set it to be 1 if the 2nd nearest neighbour sites are equally likely as the nearest sites to place the new cell
    const float P_SCALE_3RD = 0.3;  // set it to be 1 if the 3nd nearest neighbour sites are equally likely as the nearest sites to place the new cell
    bool * permit_nb26_d;
    short * ptr_nb26_d;
    cudaMallocManaged(&ptr_nb26_d, nb26_size*3*sizeof(short));
    cudaMallocManaged(&permit_nb26_d, nb26_size*sizeof(bool));
    vector<vector<int>> nb26Shu = nb26;
    random_shuffle ( nb26Shu.begin(), nb26Shu.end() );
    for (int ss = 0; ss < nb26_size; ss ++)
    {
        *(ptr_nb26_d+ss*3) = nb26Shu[ss][0];
        *(ptr_nb26_d+ss*3+1) = nb26Shu[ss][1];
        *(ptr_nb26_d+ss*3+2) = nb26Shu[ss][2];

        // update permit according to probabilities
        *(permit_nb26_d+ss) = true;
        float ra01_permit_nb = dis(gen);
        short nb_level = nb26Shu[ss][0]*nb26Shu[ss][0] + nb26Shu[ss][1]*nb26Shu[ss][1] + nb26Shu[ss][2]*nb26Shu[ss][2];
        if ((nb_level == 2 && ra01_permit_nb > P_SCALE_2ND) || (nb_level == 3 && ra01_permit_nb > P_SCALE_3RD))
            *(permit_nb26_d+ss) = false;
    }

    // -------------------------------------------------------
    // [6] call kernel
    // -------------------------------------------------------
    cout << "[6] call growth kernel : growth_random_kernel" << endl;
    int threadsPerBlock, blocksPerGrid;
    threadsPerBlock = NTHR;
    if (n_sites_valid < threadsPerBlock)
    {
        threadsPerBlock = n_sites_valid;
        blocksPerGrid = 1;
    }
    else
    {
        blocksPerGrid = ceil(float(n_sites_valid)/float(threadsPerBlock));
    }
    cout << "     KERNEL: blocksPerGrid = " << blocksPerGrid << ", threadsPerBlock = " << threadsPerBlock << endl;


    // ... prepare random numbers for growth probability
    float * ra01_cp, * ptr_ra_cp;
    cudaError_t r4 = cudaMallocManaged(&ra01_cp, n_sites_valid*sizeof(float));
    ptr_ra_cp = ra01_cp;
    for (int nra = 0; nra < n_sites_valid; ++ nra) {
        * ptr_ra_cp = dis(gen);
        ptr_ra_cp ++;
    }

    growth_random_kernel<<<blocksPerGrid, threadsPerBlock>>>(tumour, clone, sites_valid, sites_valid_pcopy,
                                                             n_sites_valid, sites_new,
                                                             permit_nb26_d, ptr_nb26_d, ra01_cp);
    cudaDeviceSynchronize();
    cudaFree(ra01_cp);

    // -------------------------------------------------------
    // [7] add new sites to vecCell according to sites_new
    // -------------------------------------------------------
    cout << "[7] fetch new valid sites & create new cells & update cell ids" << endl;
    int cid = vecCell.size();
    vector<int> vecNewCellID;
    for (int is = 0; is < n_sites_valid; is++)
    {
        vector<int> loc {*(sites_new+is*3), *(sites_new+is*3+1), *(sites_new+is*3+2)};

        if (loc[0] < 0 || loc[1] < 0 || loc[2] < 0) // not growing & dividing
        {
            continue;
        }

        // ... create new cell
        bool alive = true;
        Cell c = Cell(cid, alive, loc);
        vecCell.push_back(c);
        vecNewCellID.push_back(cid);
        vector<int> loc_p {*(sites_valid+is*3), *(sites_valid+is*3+1), *(sites_valid+is*3+2)};
        vecNewCellID.push_back(
            *(cellids + loc_p[0]*N*N + loc_p[1]*N + loc_p[2])
        );   // also the parent cell id

        // ... update cellids
        *(cellids + loc[0]*N*N + loc[1]*N + loc[2]) = cid;

        cid ++;
    }

    // -------------------------------------------------------
    // [8] accumulate driver events in new cells
    // -------------------------------------------------------
    if (typeCloneCreate == "emergeRCC")
        emerge_subclones_rcc_uponProlif(clone, vecCell, vecNewCellID, umapCloneEvents, umapCloneEventsOrder);

    // -------------------------------------------------------
    // [9] free allocated memory
    // -------------------------------------------------------
    cout << "[9] de-allocate memory" << endl;
    cudaFree(ptr_nb26_d);
    cudaFree(permit_nb26_d);
    cudaFree(ptr_ra);
    cudaFree(ptr_ra_cp);
    cudaFree(sites_valid);
    cudaFree(sites_new);
}

// TUMOUR GROWTH growth2_volume()
void growth2_volume(char * tumour, int * cellids, long * clone, vector<Cell> & vecCell,
             unordered_map <long, vector<string>> &umapCloneEvents,
             unordered_map <long, long> &umapCloneEventsOrder)
{
    // -------------------------------------------------------
    // [1] shuffle vecCell
    // -------------------------------------------------------
    cout << "[1] shuffle vecCell : random_shuffle" << endl;

    vector<Cell> vecCellShu = vecCell;
    random_shuffle ( vecCellShu.begin(), vecCellShu.end() );

    // -------------------------------------------------------
    // [2] check cell death & the number of alive cells
    // -------------------------------------------------------
    cout << "[2] check cell death & the number of alive cells" << endl;
    vector<Cell> vecCellShuValid;
    // ... prepare random numbers for cell death probability
    int n_sites_is_cell = vecCellShu.size();
    float * ra01, * ptr_ra;
    cudaError_t rrrr = cudaMallocManaged(&ra01, n_sites_is_cell*sizeof(float));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);   //uniform distribution between 0 and 1
    ptr_ra = ra01;
    for (int nra = 0; nra < n_sites_is_cell; ++ nra) {
         * ptr_ra = dis(gen);
         ptr_ra ++;
    }
    // ... push_back Cells into the vecCellShuValid vector
    ptr_ra = ra01;
    for (vector<Cell>::iterator it_C = vecCellShu.begin(); it_C != vecCellShu.end(); it_C ++)
    {
        bool alive = it_C->get_alive();
        if (alive)
        {
            // consider cell death here
            float p_empty = P_EMPTY;

            if (flagTumourApop && *ptr_ra < p_empty
                && vecCellShu.size() > N_START_EMPTY)
            {
                int cid_e = it_C->get_id();
                vecCell[cid_e].set_alive(false);
                vector<int> loc_e = it_C->get_loc();
                *(tumour+loc_e[0]*N*N+loc_e[1]*N+loc_e[2]) = 'd';
                ptr_ra ++;

                continue;
            }

            // check if it's proliferative!!!
            vector<int> loc_c = it_C->get_loc();
            if (*(tumour+loc_c[0]*N*N+loc_c[1]*N+loc_c[2]) == 'p')
            {

                // check the P_COPY here
                // ... assign P_COPY * scale_factor to this site
                float scale_factor = 1;

                if (typeCloneCreate == "emergeRCC")
                {
                    // find the cloneEvents in this clone
                    long subcl = *(clone  + loc_c[0]*N*N + loc_c[1]*N + loc_c[2]);
                    vector<string> cloneEvents = umapCloneEvents[subcl];

                    // ... saturated model
                    if (typeDriverAdvantage == 's')
                    {
                        for (vector<string>::iterator it_v = cloneEvents.begin(); it_v != cloneEvents.end(); it_v++)
                        {
                            string event = *(it_v);
                            float event_scale_factor = umapDriverProlRCC.at(event);
                            if (event_scale_factor > scale_factor)
                                scale_factor = event_scale_factor;
                        }
                    }

                    // ... additive model
                    if (typeDriverAdvantage == 'a')
                    {
                        float total_selection_coef = P_COPY;
                        for (vector<string>::iterator it_v = cloneEvents.begin(); it_v != cloneEvents.end(); it_v++)
                        {
                            string event = *(it_v);
                            float event_selection_coef = umapDriverProlAdditiveRCC.at(event);
                            total_selection_coef += event_selection_coef;

                        }
                        scale_factor = total_selection_coef / P_COPY;
                    }
                }

                if (dis(gen) < P_COPY * scale_factor)
                    vecCellShuValid.push_back(*it_C);
            }
        }
    }
    cudaFree(ra01);

    // -------------------------------------------------------
    // [3] allocate memory for valid sites
    // -------------------------------------------------------
    int n_sites_valid = vecCellShuValid.size();

    //cout << "[3] allocate memory for valid sites : cudaMallocManaged" << endl;
    cout << "     n_sites_valid = " << n_sites_valid << endl;

    // -------------------------------------------------------
    // [5] shuffle neighbourhood
    // -------------------------------------------------------
    cout << "[5] shuffle neighbourhood nb26 : random_shuffle" << endl;
    const float P_SCALE_2ND = 0.3;  // set it to be 1 if the 2nd nearest neighbour sites are equally likely as the nearest sites to place the new cell
    const float P_SCALE_3RD = 0.3;  // set it to be 1 if the 3nd nearest neighbour sites are equally likely as the nearest sites to place the new cell
    bool * permit_nb26_d;    // update 20191114: consider probability of being selected for 2nd/3rd nearest neighbours
    short * ptr_nb26_d;

    cout << "[6] volume growth : duplication & push voxels outward" << endl;

    int i,j,k, di, dj, dk;
    int cid = vecCell.size();
    vector<int> vecNewCellID;

    for (vector<Cell>::iterator it_C = vecCellShuValid.begin(); it_C != vecCellShuValid.end(); it_C ++)
    {
        vector<int> loc_c = it_C->get_loc();
        i = loc_c[0]; j = loc_c[1]; k = loc_c[2];

        cudaMallocManaged(&ptr_nb26_d, nb26_size*3*sizeof(short));
        cudaMallocManaged(&permit_nb26_d, nb26_size*sizeof(bool));
        vector<vector<int>> nb26Shu = nb26;

        random_shuffle ( nb26Shu.begin(), nb26Shu.end() );
        for (int ss = 0; ss < nb26_size; ss ++)
        {
            *(ptr_nb26_d+ss*3) = nb26Shu[ss][0];
            *(ptr_nb26_d+ss*3+1) = nb26Shu[ss][1];
            *(ptr_nb26_d+ss*3+2) = nb26Shu[ss][2];

            // update permit according to probabilities
            *(permit_nb26_d+ss) = true;
            float ra01_permit_nb = dis(gen);
            short nb_level = nb26Shu[ss][0]*nb26Shu[ss][0] + nb26Shu[ss][1]*nb26Shu[ss][1] + nb26Shu[ss][2]*nb26Shu[ss][2];
            if ((nb_level == 2 && ra01_permit_nb > P_SCALE_2ND) || (nb_level == 3 && ra01_permit_nb > P_SCALE_3RD))
                *(permit_nb26_d+ss) = false;
        }

        // find the direction to place daughter voxel -- THE ONE orientation giving shortest distance to surface
        if (true)
        {
            short num_checked = 0, num_check_max = 5;  // Waclaw 2015 paper checked 10 directions
            short di_, dj_, dk_;    // orientation with shortest distance
            float dist_ = N;
            for (short inb = 0; inb < nb26_size; inb ++)
            {
                di = *(ptr_nb26_d+3*inb);
                dj = *(ptr_nb26_d+3*inb+1);
                dk = *(ptr_nb26_d+3*inb+2);

                // find how far away from surface
                int n_steps = 0;
                bool flagNotYetSurface = true;
                bool flagInvalidDirection = false;  // this refers to directions leading to a boundary or outside the domain
                while (flagNotYetSurface)
                {
                    n_steps ++;

                    // avoid going outside the simulated domain
                    if (i+n_steps*di < 0 || i+n_steps*di > N-1 ||
                        j+n_steps*dj < 0 || j+n_steps*dj > N-1 ||
                        k+n_steps*dk < 0 || k+n_steps*dk > N-1)
                    {
                        flagNotYetSurface = false;
                        flagInvalidDirection = true;
                    }

                    // check if the status is not tumour
                    char status_next = *(tumour + (i+di*n_steps)*N*N + (j+dj*n_steps)*N + k+dk*n_steps);
                    if (status_next != 'p')  // not any more a tumour site
                        flagNotYetSurface = false;

                    // avoid going against of boundary
                    if (status_next == 'b')
                        flagInvalidDirection = true;
                }

                // continue to explore other directions is this direction is invalid
                if (flagInvalidDirection)
                    continue;

                float dist_this = n_steps*sqrt( di*di + dj*dj + dk*dk );
                if (dist_this < dist_)
                {
                    dist_ = dist_this;
                    di_ = di; dj_ = dj; dk_ = dk;
                }

                num_checked ++;
                if (num_checked > num_check_max)
                    break;

            }
            di = di_; dj = dj_; dk = dk_;
        }

        // find the cells to be pushed
        // ... collect sites until hitting an empty site
        int i2 = i+di, j2 = j+dj, k2 = k+dk;

        // ... create new cell
        bool alive = true;
        vector<int> loc_newcell {i2,j2,k2};
        Cell c = Cell(cid, alive, loc_newcell);
        vecCell.push_back(c);
        vecNewCellID.push_back(cid);
        vecNewCellID.push_back(
            *(cellids + loc_c[0]*N*N + loc_c[1]*N + loc_c[2])
        );   // also the parent cell id

        // ... push other cells outward
        char status1 = *(tumour + i2*N*N + j2*N + k2);
        long subcl1, cell1, subcl2, cell2;
        if (status1 == 'p')
        {
            subcl1 = *(clone  + i2*N*N + j2*N + k2);
            cell1 = *(cellids + i2*N*N + j2*N + k2);
        }
        while (status1 == 'p')
        {
            i2 += di; j2 += dj; k2 += dk;
            if (i2 < 0 || i2 > N-1 ||
                j2 < 0 || j2 > N-1 ||
                k2 < 0 || k2 > N-1)
                break;   // outside: avoid overflow error

            //cout << "... checkpoint #2.1 ..." << endl;
            char status2 = *(tumour + i2*N*N + j2*N + k2);
            // ... copy status2 to 1
            if (status2 == 'p')
            {
                subcl2 = *(clone  + i2*N*N + j2*N + k2);
                cell2 = *(cellids + i2*N*N + j2*N + k2);
            }

            // ... set status1, subcl1 to 2
            *(tumour + i2*N*N + j2*N + k2) = status1;
            *(clone  + i2*N*N + j2*N + k2) = subcl1;
            *(cellids  + i2*N*N + j2*N + k2) = cell1;
            vecCell[cell1].set_loc({i2,j2,k2});

            if (status2 == 'p')
            {
                status1 = status2;
                subcl1 = subcl2;
                cell1 = cell2;
            }
            else
            {
                break;
            }

        }

        // ... lastly update status for the newcell
        *(cellids + loc_newcell[0]*N*N + loc_newcell[1]*N + loc_newcell[2]) = cid;
        *(tumour + loc_newcell[0]*N*N + loc_newcell[1]*N + loc_newcell[2]) = *(tumour + i*N*N + j*N + k);
        *(clone + loc_newcell[0]*N*N + loc_newcell[1]*N + loc_newcell[2]) = *(clone  + i*N*N + j*N + k);

        cid ++;

        cudaFree(ptr_nb26_d);
        cudaFree(permit_nb26_d);


    }

    // -------------------------------------------------------
    // [8] accumulate driver events in new cells
    // -------------------------------------------------------
    emerge_subclones_rcc_uponProlif(clone, vecCell, vecNewCellID, umapCloneEvents, umapCloneEventsOrder);


    // -------------------------------------------------------
    // [9] free allocated memory
    // -------------------------------------------------------
    cout << "[8] de-allocate memory" << endl;
    cudaFree(ptr_ra);
}


// TUMOUR NECROSIS necrosis2() -- gpu version
void necrosis2(char * tumour, int * cellids, long * clone, vector<Cell> & vecCell)
{
    // update surface voxels
    vector<vector<int>> vecVoxSurf;
    update_surface2(tumour, vecVoxSurf);    // get a vector of cells located at the tumour surface
    random_shuffle ( vecVoxSurf.begin(), vecVoxSurf.end() );

    // a boolean data structure to record which voxel is necrotic
    bool * near_surface;
    cudaError_t r   = cudaMallocManaged(&near_surface, N*N*N*sizeof(bool));

    // make smaller amount of data for gpu
    vector<vector<int>> vecValidSites;
    for (int i = 0; i < N; i ++)
    {
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k ++)
            {
                char status = *(tumour + i*N*N + j*N + k);
                if (status == 'p' || status == 'n')
                {
                    vector<int> vox {i,j,k};
                    vecValidSites.push_back(vox);
                }
            }
        }
    }
    int n_sites_valid = vecValidSites.size();
    int * sites_valid;
    cudaError_t r2 = cudaMallocManaged(&sites_valid, n_sites_valid*3*sizeof(int));
    for (int ss = 0; ss < n_sites_valid; ss ++)
    {
        *(sites_valid+ss*3) = vecValidSites[ss][0];
        *(sites_valid+ss*3+1) = vecValidSites[ss][1];
        *(sites_valid+ss*3+2) = vecValidSites[ss][2];
    }

    // device acceptable format for vecVoxSurf
    int n_voxSurf = vecVoxSurf.size();
    n_voxSurf = min(500, n_voxSurf);
    short * ptr_voxSurf_d;
    cudaMallocManaged(&ptr_voxSurf_d, n_voxSurf*3*sizeof(short));
    for (int ss = 0; ss < n_voxSurf; ss ++)
    {
        *(ptr_voxSurf_d+ss*3) = vecVoxSurf[ss][0];
        *(ptr_voxSurf_d+ss*3+1) = vecVoxSurf[ss][1];
        *(ptr_voxSurf_d+ss*3+2) = vecVoxSurf[ss][2];
    }

    int threadsPerBlock, blocksPerGrid;
    threadsPerBlock = NTHR;
    if (n_sites_valid < threadsPerBlock)
    {
        threadsPerBlock = n_sites_valid;
        blocksPerGrid = 1;
    }
    else
    {
        blocksPerGrid = ceil(float(n_sites_valid)/float(threadsPerBlock));
    }
    cout << "     KERNEL: blocksPerGrid = " << blocksPerGrid << ", threadsPerBlock = " << threadsPerBlock << endl;

    necrosis_kernel<<<blocksPerGrid, threadsPerBlock>>>(near_surface, n_sites_valid, sites_valid, n_voxSurf, ptr_voxSurf_d);
    cudaDeviceSynchronize();

    // update necrosis
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);   //uniform distribution between 0 and 1
    for (int i = 0; i < N; i ++)
    {
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k ++)
            {
                char status = *(tumour + i*N*N + j*N + k);
                if (status == 'p' || status == 'n')
                {
                    bool is_near_surface = *(near_surface+i*N*N+j*N+k);
                    if (is_near_surface == false)
                    {
                        // check probability of necrosis
                        float random01 = dis(gen);
                        if (random01 < P_NECROSIS)
                        {
                            int cid = *(cellids+i*N*N+j*N+k);
                            vecCell[cid].set_alive(false);
                            *(tumour+i*N*N+j*N+k) = 'd';

                            // !!! Update proliferative status for the surrounding cells of dying cell !!!
                            int dx, dy, dz;
                            for (int ss = 0; ss < nb26_size; ss ++)
                            {
                                dx = nb26[ss][0];
                                dy = nb26[ss][1];
                                dz = nb26[ss][2];
                                vector<int> loc_n {i+dx, j+dy, k+dz};
                                if ( *(tumour+loc_n[0]*N*N+loc_n[1]*N+loc_n[2]) == 'n' )    // non-proliferative
                                    *(tumour+loc_n[0]*N*N+loc_n[1]*N+loc_n[2]) = 'p';
                            }
                        }
                    }
                }
            }
        }
    }

    cudaFree(near_surface);
    cudaFree(sites_valid);
    cudaFree(ptr_voxSurf_d);
}

void update_surface2(char * tumour, vector<vector<int>> & vecVoxSurf)
{
    // update surface voxels
    bool * surface;
    cudaError_t r   = cudaMallocManaged(&surface, N*N*N*sizeof(bool));

    short * ptr_nb26_d;
    cudaMallocManaged(&ptr_nb26_d, nb26_size*3*sizeof(short));
    vector<vector<int>> nb26Shu = nb26;
    random_shuffle ( nb26Shu.begin(), nb26Shu.end() );
    for (int ss = 0; ss < nb26_size; ss ++)
    {
        *(ptr_nb26_d+ss*3) = nb26Shu[ss][0];
        *(ptr_nb26_d+ss*3+1) = nb26Shu[ss][1];
        *(ptr_nb26_d+ss*3+2) = nb26Shu[ss][2];
    }

    int threadsPerBlock, blocksPerGrid;
    threadsPerBlock = NTHR;
    blocksPerGrid = ceil(float(N*N*N)/float(threadsPerBlock));
    cout << "     KERNEL: blocksPerGrid = " << blocksPerGrid << ", threadsPerBlock = " << threadsPerBlock << endl;

    update_surface_kernel<<<blocksPerGrid, threadsPerBlock>>>(tumour, surface, ptr_nb26_d);
    cudaDeviceSynchronize();

    // update vecVoxSurf
    for (int i = 0; i < N; i ++)
    {
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k ++)
            {
                bool is_surface_voxel = *(surface+i*N*N+j*N+k);
                if (is_surface_voxel == true)
                {
                    vector<int> vox {i,j,k};
                    vecVoxSurf.push_back(vox);
                }
            }
        }
    }

    cudaFree(surface);
    cudaFree(ptr_nb26_d);
}

// acquire driver mutations/SCNAs upon proliferation (typeCloneCreate == "emergeRCC")
void emerge_subclones_rcc_uponProlif(long * clone, vector<Cell> &vecCell, vector<int> vecNewCellID,
                          unordered_map <long, vector<string>> &umapCloneEvents, unordered_map <long, long> &umapCloneEventsOrder)
{
    // find the number of subclones in umapCloneEvents
    long cnt_subcl = 0;
    for (unordered_map<long, vector<string>>::iterator it_u = umapCloneEvents.begin(); it_u != umapCloneEvents.end(); it_u++)
    {
        cnt_subcl += 1;
    }

    for (vector<int>::iterator it_i = vecNewCellID.begin(); it_i != vecNewCellID.end(); it_i ++)
    {
        int cell_id = *it_i;
        vector<int> loc = vecCell[cell_id].get_loc();

        // [1]
        long subcl_parent = *(clone  + loc[0]*N*N + loc[1]*N + loc[2]);
        vector<string> cloneEvents_parent = umapCloneEvents[subcl_parent];

        // [2]
        vector<string> cloneEvents = cloneEvents_parent;
        for (short imuta = 0; imuta < N_EVENT_DRIVER_RCC; imuta ++)
        {
            // check not already in the cloneEvents_parent
            if (find(cloneEvents_parent.begin(), cloneEvents_parent.end(), arrMutaDriverNameRCC[imuta])
                == cloneEvents_parent.end() )
            {
                float ra = rand()/(float)RAND_MAX;

                // scale factor for P_EVENT_DRIVER_RCC
                float scale_factor_emerge = umapDriverEmerRCC.at(arrMutaDriverNameRCC[imuta]);
                // ... check if mutations increasing SCNA freq are in the cloneEvents_parent
                if (arrMutaDriverNameRCC[imuta].find("loss") != string::npos ||
                    arrMutaDriverNameRCC[imuta].find("gain") != string::npos)   // only if it's SCNAs
                {
                    for (short jmuta = 0; jmuta < N_MUTA_INCREASE_SCNA_RCC; jmuta ++)
                    {
                        if ( find(cloneEvents_parent.begin(), cloneEvents_parent.end(), arrMutaIncreaseScnaRCC[jmuta])
                        != cloneEvents_parent.end()  )
                            scale_factor_emerge = 1;
                    }
                }

                if (ra < P_EVENT_DRIVER_RCC_UPON_PROLIF*scale_factor_emerge)
                    cloneEvents.push_back(arrMutaDriverNameRCC[imuta]);
            }
        }

        // [3]
        if (cloneEvents.size() > cloneEvents_parent.size())
        {
            *(clone  + loc[0]*N*N + loc[1]*N + loc[2]) = cnt_subcl;
            // ... add the subcl to the umapCloneEvents
            umapCloneEvents[cnt_subcl] = cloneEvents;
            // ... add the parent-child relationship to the umapCloneEventsOrder
            umapCloneEventsOrder[cnt_subcl] = subcl_parent;

            cnt_subcl ++;
        }
    }
}

// This function writes cell coordinates into a '.txt' format
void writeDynamics(char * tumour, long *clone, int *cellids,
                   unordered_map<long, vector<string>> &umapCloneEvents,
                   unordered_map <long, long> &umapCloneEventsOrder, int t_now,
               bool flagWriteCoord, bool flagIsFinal)
{
    /*
        This function saves the following output files:
        (1) 3D cell positions of cells located at tumour surface over time
        (2) 2D cell positions (z = 0) over time
        (3) Prevalence (i.e. number of cells) of individual events over time
        (4) Prevalence (i.e. number of cells) of individual clones over time
    */

    // ==================== cancer cells ========================
    // header: t, CellID, LineageID, x, y, z
    ofstream nodeFile;
    string nodeFileName;
    stringstream PROC_ID_SS; PROC_ID_SS << PROC_ID;
    nodeFileName = "PID_" + PROC_ID_SS.str() + "_cellDynamics.txt";

    nodeFile.open(nodeFileName, ios::app | ios::binary);
    if (t_now == 0)
        nodeFile << "t\tCellID\tLineageID\tx\ty\tz" << endl;

    // ... 2D XY plane
    ofstream nodeFileXY;
    string nodeFileXYName;
    nodeFileXYName = "PID_" + PROC_ID_SS.str() + "_cellDynamicsXY.txt";

    nodeFileXY.open(nodeFileXYName, ios::app | ios::binary);
    if (t_now == 0)
        nodeFileXY << "t\tCellID\tLineageID\tx\ty\tz" << endl;

    // ==========================================================

    if (t_now > T-FOUTPUT_SURF)
        flagIsFinal = true;

    // ================ clones (size over time) =================
    // header: t, cloneEvent, nCell
    ofstream nodeFile3a;
    string nodeFileName3a;
    nodeFileName3a = "PID_" + PROC_ID_SS.str() + "_eventSizeOverTime.txt";

    nodeFile3a.open(nodeFileName3a, ios::app | ios::binary);
    if (t_now == 0)
        nodeFile3a << "t\tRCCevent\tnCell" << endl;

    // header: t, LineageID, nCell
    ofstream nodeFile3b;
    string nodeFileName3b;
    nodeFileName3b = "PID_" + PROC_ID_SS.str() + "_cloneSizeOverTime.txt";

    nodeFile3b.open(nodeFileName3b, ios::app | ios::binary);
    if (t_now == 0)
        nodeFile3b << "t\tLineageID\tnCell" << endl;

    // ... a couple unordered_map to collect information
    unordered_map<string, long> umapEventNumCell;
    unordered_map<long, long> umapCloneNumCell;

    // ==========================================================

    for (int i = 0; i < N; i ++)
    {
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k ++)
            {
                int ivox = i*N*N + j*N + k;
                int cid = *(cellids + ivox);
                char status = *(tumour + ivox);
                long subcl = *(clone  + ivox);

                // clone: collect number of cells in clone & with events
                if (typeCloneCreate == "emergeRCC" &&
                    (status == 'p' || status == 'n') )
                {
                    // .. by clone
                    if (umapCloneNumCell.find(subcl) == umapCloneNumCell.end())
                        umapCloneNumCell[subcl] = 0;
                    umapCloneNumCell[subcl] ++;
                    // .. by event
                    vector<string> cloneEvents = umapCloneEvents[subcl];
                    for (vector<string>::iterator it_v = cloneEvents.begin(); it_v != cloneEvents.end(); it_v++)
                    {
                        string event = *(it_v);
                        if (umapEventNumCell.find(event) == umapEventNumCell.end())
                            umapEventNumCell[event] = 0;
                        umapEventNumCell[event] ++;
                    }
                }

                // .. XY plane: k == N/2 layer of cancer cells
                if (k == N/2 && status != 'e')
                {
                    if (status == 'b') // confinement or barriers
                    {
                        cid = -1;
                        subcl = -1;
                    }
                    if (status == 'd') // dead regions
                    {
                        cid = -2;
                        subcl = -2;
                    }

                    stringstream sst, ssid, sslid, ssx, ssy, ssz;
                    sst << t_now;
                    ssid << cid;
                    sslid << subcl;
                    ssx << i;
                    ssy << j;
                    ssz << k;

                    if (flagWriteCoord)
                        nodeFileXY << sst.str() << "\t" << ssid.str() << "\t" << sslid.str() << "\t"
                    << ssx.str() << "\t" << ssy.str() << "\t" << ssz.str() << "\t" << endl;
                }
            }
        }
    }

    // write tumour surface
    vector<vector<int>> vecVoxSurf;
    update_surface2(tumour, vecVoxSurf);
    for (vector<vector<int>>::iterator it_v = vecVoxSurf.begin(); it_v != vecVoxSurf.end(); it_v ++)
    {
        vector<int> voxSurf = *it_v;
        int i = voxSurf[0], j = voxSurf[1], k = voxSurf[2];
        int ivox = i*N*N + j*N + k;
        int cid = *(cellids + ivox);
        char status = *(tumour + ivox);
        long subcl = *(clone  + ivox);

        // surface layer of cancer cells
        stringstream sst, ssid, sslid, ssx, ssy, ssz;
        sst << t_now;
        ssid << cid;
        sslid << subcl;
        ssx << i;
        ssy << j;
        ssz << k;

        if (flagWriteCoord)
            nodeFile << sst.str() << "\t" << ssid.str() << "\t" << sslid.str() << "\t"
        << ssx.str() << "\t" << ssy.str() << "\t" << ssz.str() << "\t" << endl;
    }

    nodeFile.close();
    nodeFileXY.close();

    // ==================== clone size over time ===================
    if (typeCloneCreate == "emergeRCC")
    {
        // umapCloneNumCell
        // header: time, RCCevent, ncell
        for (unordered_map<string, long>::iterator it_u = umapEventNumCell.begin(); it_u != umapEventNumCell.end(); it_u++)
        {
            string event = it_u -> first;
            long ncell = it_u -> second;
            stringstream sst, ssnc;
            sst << t_now;
            ssnc << ncell;
            nodeFile3a << sst.str() << "\t" << event << "\t" << ssnc.str() << endl;
        }

        // umapEventNumCell
        // header: time, lineage, ncell
        for (unordered_map<long, long>::iterator it_u = umapCloneNumCell.begin(); it_u != umapCloneNumCell.end(); it_u++)
        {
            long subcl = it_u -> first;
            long ncell = it_u -> second;
            stringstream sst, sssc, ssnc;
            sst << t_now;
            sssc << subcl;
            ssnc << ncell;
            nodeFile3b << sst.str() << "\t" << sssc.str() << "\t" << ssnc.str() << endl;
        }
        nodeFile3a.close();
        nodeFile3b.close();
    }

    // ==================== clone information ======================
    if (typeCloneCreate == "emergeRCC" && flagIsFinal == true)
    {
        // umapCloneEvents
        // header: LineageID, cloneEvents
        ofstream nodeFile4;
        string nodeFileName4;
        nodeFileName4 = "PID_" + PROC_ID_SS.str() + "_cloneEvents.txt";
        nodeFile4.open(nodeFileName4, ios::app | ios::binary);
        nodeFile4 << "LineageID\tcloneEvents" << endl;

        for (unordered_map<long, vector<string>>::iterator it_u = umapCloneEvents.begin(); it_u != umapCloneEvents.end(); it_u ++)
        {
            long subcl = it_u -> first;
            //cout << subcl << endl;
            vector<string> cloneEvents = it_u -> second;
            stringstream sslid;
            sslid << subcl;
            nodeFile4 << sslid.str() << "\t";
            for (vector<string>::iterator it_v = cloneEvents.begin(); it_v != cloneEvents.end(); it_v ++)
            {
                nodeFile4 << *(it_v) << "\t";
                //cout << *(it_v) << "\t";
            }
            nodeFile4 << endl;
            //cout << endl;
        }
        nodeFile4.close();

        // umapCloneEventsOrder
        // header: cloneChild, cloneParent
        ofstream nodeFile5;
        string nodeFileName5;
        nodeFileName5 = "PID_" + PROC_ID_SS.str() + "_cloneEventsOrder.txt";
        nodeFile5.open(nodeFileName5, ios::app | ios::binary);
        nodeFile5 << "cloneChild\tcloneParent" << endl;

        for (unordered_map<long, long>::iterator it_u = umapCloneEventsOrder.begin(); it_u != umapCloneEventsOrder.end(); it_u ++)
        {
            long subclChild = it_u -> first;
            long subclParent = it_u -> second;
            stringstream sslid, sslid2;
            sslid << subclChild; sslid2 << subclParent;
            nodeFile5 << sslid.str() << "\t" << sslid2.str() << endl;
        }
        nodeFile5.close();
    }
}
