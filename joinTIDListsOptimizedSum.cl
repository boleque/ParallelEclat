
__kernel void joinTIDListsOptimizedSum(  int size,
                                         __global uint* prefixSumList,
                                         __global uint* upTIDList, 
                                         __global uint* downTIDList,
                                         __global uint* joinedTIDList,
										 __local  uint* localList
                                      )
    {
        uint global_id        = get_global_id(0);
        uint local_id         = get_local_id(0);
        uint work_group_id    = get_group_id(0);
        uint work_group_size  = get_local_size(0);
        uint work_group_shift = work_group_id * size;

        uint maxval = 0; 

        localList[global_id] = upTIDList[global_id] & downTIDList[global_id];

        event_t ev_copyTIDlists = async_work_group_copy(&joinedTIDList[work_group_size * work_group_id], localList, work_group_size, 0);
        wait_group_events(1, &ev_copyTIDlists);

        do
        {
            uint bitCount = popcount( localList[work_group_shift + local_id] );
            
            uint prefixSum = work_group_scan_inclusive_add( bitCount );
            prefixSumList[work_group_shift + local_id] = prefixSum + maxval;
 
            maxval += work_group_broadcast( prefixSum + bitCount, work_group_size - 1 );
 
            work_group_shift += work_group_size;
        }
        while(work_group_shift < (work_group_id + 1) * size);

    }