import numba
import numpy as np

#function to find the highest PID for particles
@numba.jit(nopython=True)
def maxHistoryPDGID(idxList_contents, idxList_starts, idxList_stops, pdgID_contents, pdgID_starts, pdgID_stops, motherIdx_contents, motherIdx_starts, motherIdx_stops):
    maxPDGID_array = np.ones(len(idxList_starts),np.int32)*-1
    for i in range(len(idxList_starts)):
        if idxList_starts[i]==idxList_stops[i]:
            continue
            
        idxList = idxList_contents[idxList_starts[i]:idxList_stops[i]]
        pdgID = pdgID_contents[pdgID_starts[i]:pdgID_stops[i]]
        motherIdx = motherIdx_contents[motherIdx_starts[i]:motherIdx_stops[i]]
    
        idx = idxList[0]
        maxPDGID = -1
        while idx>-1:
            pdg = pdgID[idx]
            maxPDGID = max(maxPDGID, abs(pdg))
            idx = motherIdx[idx]
        maxPDGID_array[i] = maxPDGID
    return maxPDGID_array


