### multi grid
1 0            ;grid levels, number of coarsening step to reach the finest grid
### directed coarsening
2 2 0          ;additional coarsening in 'i', 'j', 'k' direction
### number of processors
1              ;number of processors
### mapping strategy
7              ;mapping strategy (9 = manually, else one processor)
### processor info
1 1 1 : 1 ;2 3 4         ;proc. in i, j, k dir., new processor (block 1)
;1 1 1 : 1
;1 1 1 0        ;proc. in i, j, k dir., new processor (block 2)
;1 1 1 0        ;proc. in i, j, k dir., new processor (block 3)
### number of flow regions
1
;### flow region description
1         ;flow region for block #1, #2 and #3
### moving grids
0               ;moving 1/0; elliptic only i-k-planes !!!
### fluid-structure interaction
0
### smooth grid
0               ;grid smoothing 1/0
### highmem fast calculation
0       ;high mem usage fast calculation 1/0
### precision (bytes)
8
### volume check
1              ;1 = on, 0 = off
### connectivity accuracy
1e-5
### create bc files
0              ;1 = create, 0 = do not create
### plot grid geometries
unpart         ;[none | unpart | part | explode]
               ; unpart - geometric block structure
               ; part   - partitioned block structure with scalar P indicating assigned processor 
               ; explode- experimental exploded view (only works for clustered block distribution)
### monitoring point
1 2 2 2        ;block, i, j, k
### turbulence statistics
0              ;input number for turbulence statistics (memory allocation)
               ; 0 = no memory for turbulence statistics is allocated
               ; 1 = memory for 1st and 2nd order moments is allocated 
               ; 2 = memory for 1st, 2nd, 3rd and 4th order moments is allocated 
### number of chemical species
1
### number of chemical reactions
0
### number of comb. prog. variables
0
### allocation
isotherm unsteady laminar       ;keywords are:
                                ;    isotherm   or  temperature
                                ;    steady     or  unsteady
                                ;    laminar    or  turbulent
                                                ;  if turbulent____ keps
                                                ;                | 
                                                ;                |__les
                                                ;                | 
                                                ;                |__kzf
                                ;    dlmin  --> calculation of wall distance based on transport eq.
