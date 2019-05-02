__kernel void joinTIDListsNaivePrefixSum(  int size,
                                          __global uint* prefixSumList,
                                          __global uint* upTIDList, 
                                          __global uint* downTIDList,
                                          __global uint* joinedTIDList,
                                          __local uint*  localList
                                        )
    {
		uint global_id        = get_global_id(0);
        uint local_id         = get_local_id(0);
        uint work_group_id    = get_group_id(0);
        uint work_group_size  = get_local_size(0);
        uint work_group_shift = work_group_id * size;

        local uint maxval;

        if( local_id == 0 )  maxval = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        localList[global_id] = upTIDList[global_id] & downTIDList[global_id];
       
        event_t ev_copyTIDlists = async_work_group_copy(&joinedTIDList[work_group_size * work_group_id], localList, work_group_size, 0);
        wait_group_events(1, &ev_copyTIDlists);    

        do
        {
		    uint prefixSum = 0;
		    for(int indx = 0; indx <= local_id; indx++)
			    prefixSum += popcount( localList[indx] );
         
		    prefixSumList[work_group_shift + local_id] = prefixSum + maxval;
            prefixSum += popcount(localList[work_group_shift + local_id]);

            if( local_id == work_group_size - 1 )  
                maxval += prefixSum;
            barrier(CLK_LOCAL_MEM_FENCE);

            work_group_shift += work_group_size;
        }
        while(work_group_shift < (work_group_id + 1) * size);

    }

